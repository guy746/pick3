import time
import threading
import json
import logging
from base_agent import BaseAgent

class ScoringAgent(BaseAgent):
    """Standalone agent for prioritizing and assigning work to CNC"""
    
    def __init__(self):
        super().__init__(
            name="ScoringAgent",
            subscribe_channels=['events:vision', 'events:cnc'],
            publish_channel='events:cnc'
        )
        self.tracked_objects = {}
        self.scoring_weights = {'fifo': 1.0, 'position': 0.0, 'urgency': 0.0}
        self.lock = threading.Lock()
        self.pickup_zone_start = 300
        self.pickup_zone_end = 375
        self.pickup_zone_center = (self.pickup_zone_start + self.pickup_zone_end) / 2
        
    def handle_message(self, channel, message):
        """Handle incoming messages from subscribed channels"""
        try:
            event = message
            if channel == 'events:vision' and event.get('event') == 'object_detected':
                self.handle_vision_detection(event)
            elif channel == 'events:cnc' and event.get('event') == 'ready_for_assignment':
                self.handle_cnc_ready(event)
        except Exception as e:
            self.logger.error(f"Error processing message: {e}", exc_info=True)
    
    # Removed duplicate __init__ - using BaseAgent implementation
        
    def handle_vision_detection(self, event_data):
        """Handle object detection from vision agent"""
        obj_data = event_data.get('data', {})
        obj_id = obj_data.get('id')
        obj_type = obj_data.get('type')
        
        if obj_type == 'green' and obj_id:
            with self.lock:
                self.tracked_objects[obj_id] = {
                    'timestamp': event_data.get('timestamp', time.time()),
                    'position': obj_data.get('position_x', 0),
                    'lane': obj_data.get('lane', 0),
                    'area': obj_data.get('area', 1500),
                    'height': obj_data.get('height', 32.5),
                    'type': obj_type
                }
            self.logger.info(f"Tracking green object {obj_id} at {obj_data.get('position_x', 0)}mm")
    
    def handle_cnc_ready(self, event_data):
        """Handle CNC ready for assignment event"""
        cnc_id = event_data.get('data', {}).get('cnc_id', 'cnc:0')
        best_object = self.get_best_assignment()
        
        if best_object:
            assignment = {
                'event': 'pickup_assignment',
                'timestamp': time.time(),
                'data': {
                    'cnc_id': cnc_id,
                    'object_id': best_object['id'],
                    'position': best_object['position'],
                    'lane': best_object['lane'],
                    'timestamp': best_object['timestamp']
                }
            }
            self.redis.publish('events:cnc', json.dumps(assignment))
            self.logger.debug(f"Published assignment event: {assignment}")
            self.logger.info(f"Assigned {best_object['id']} to {cnc_id} (LIFO: timestamp {best_object['timestamp']:.3f})")
            
            # Publish target confirmation for assigned object only
            confirmation = {
                'event': 'target_confirmed',
                'timestamp': time.time(),
                'data': {
                    'object_id': best_object['id'],
                    'lane': best_object['lane'],
                    'position_x': best_object['position']
                }
            }
            self.redis.publish('events:scoring', json.dumps(confirmation))
            self.logger.info(f"Confirmed target {best_object['id']} in lane {best_object['lane']}")
            
            with self.lock:
                self.tracked_objects.clear()
                self.logger.debug("Cleared all tracked objects")
        else:
            self.logger.warning(f"No pickable objects available for {cnc_id}")
            self.logger.debug(f"Current tracked objects: {list(self.tracked_objects.keys())}")
    
    def get_best_assignment(self):
        """Find the best object to pick using LIFO (Last In, First Out) to give CNC maximum setup time"""
        with self.lock:
            # LIFO: Select the most recently detected pickable object (furthest from pickup zone = max setup time)
            most_recent_object = None
            most_recent_timestamp = 0
            
            for obj_id, tracked_data in self.tracked_objects.items():
                obj_data = self.redis.hget(f'object:{obj_id}', 'position_x')
                if not obj_data:
                    continue
                    
                current_position = float(obj_data)
                
                # Check if object is pickable (green objects or objects in pickup zone)
                is_pickable = False
                if tracked_data.get('type') == 'green':
                    is_pickable = True  # Green objects are always pickable
                elif self.pickup_zone_start <= current_position <= self.pickup_zone_end:
                    is_pickable = True  # Objects in pickup zone are pickable
                
                # LIFO: Select the object with the most recent timestamp (detected last = furthest from pickup)
                if is_pickable and tracked_data['timestamp'] > most_recent_timestamp:
                    most_recent_timestamp = tracked_data['timestamp']
                    most_recent_object = {
                        'id': obj_id,
                        'position': current_position,
                        'lane': tracked_data['lane'],
                        'timestamp': tracked_data['timestamp']
                    }
            
            return most_recent_object
            
            # COMMENTED OUT: Original scoring-based assignment logic
            # candidates = []
            # for obj_id, tracked_data in self.tracked_objects.items():
            #     obj_data = self.redis.hget(f'object:{obj_id}', 'position_x')
            #     if not obj_data:
            #         continue
            #         
            #     current_position = float(obj_data)
            #     # Prioritize green objects regardless of position
            #     if tracked_data.get('type') == 'green':
            #         # Create a special candidate for green objects
            #         score = self.calculate_score(obj_id, tracked_data, current_position)
            #         candidates.append({
            #             'id': obj_id,
            #             'position': current_position,
            #             'lane': tracked_data['lane'],
            #             'score': score + 1000  # Boost score for green objects
            #         })
            #     elif self.pickup_zone_start <= current_position <= self.pickup_zone_end:
            #         # Only consider non-green objects if they're in the pickup zone
            #         score = self.calculate_score(obj_id, tracked_data, current_position)
            #         candidates.append({
            #             'id': obj_id,
            #             'position': current_position,
            #             'lane': tracked_data['lane'],
            #             'score': score
            #         })
            #         
            # return max(candidates, key=lambda x: x['score']) if candidates else None
    
    # COMMENTED OUT: Original scoring calculation method - now using LIFO
    # def calculate_score(self, obj_id, tracked_data, current_position):
    #     """Calculate priority score for an object"""
    #     score = 0.0
    #     if self.scoring_weights['fifo'] > 0:
    #         time_since_detection = time.time() - tracked_data['timestamp']
    #         fifo_score = min(time_since_detection / 10.0, 1.0)
    #         score += self.scoring_weights['fifo'] * fifo_score
    #     if self.scoring_weights['position'] > 0:
    #         distance_from_center = abs(current_position - self.pickup_zone_center)
    #         max_distance = (self.pickup_zone_end - self.pickup_zone_start) / 2
    #         position_score = 1.0 - (distance_from_center / max_distance)
    #         score += self.scoring_weights['position'] * position_score
    #     if self.scoring_weights['urgency'] > 0:
    #         distance_to_exit = self.pickup_zone_end - current_position
    #         zone_length = self.pickup_zone_end - self.pickup_zone_start
    #         urgency_score = 1.0 - (distance_to_exit / zone_length)
    #         score += self.scoring_weights['urgency'] * urgency_score
    #     return score

def main():
    """Run standalone scoring agent"""
    agent = ScoringAgent()
    agent.run()

if __name__ == '__main__':
    main()
