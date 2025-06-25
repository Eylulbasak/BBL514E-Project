import math

def calculate_iou(box1, box2):
    
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
 
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    if x2_i <= x1_i or y2_i <= y1_i:
        return 0.0
    
    intersection = (x2_i - x1_i) * (y2_i - y1_i)
    
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0

class Tracker:
    def __init__(self):
        self.center_points = {}
        self.id_count = 0  
        self.frames_without_detection = {} 
        self.previous_boxes = {} 

    def update(self, objects_rect):
        
        objects_bbs_ids = []
        
        
        for id in self.center_points.keys():
            self.frames_without_detection[id] = self.frames_without_detection.get(id, 0) + 1
        
        
        matched_objects = set()
        
        for rect in objects_rect:
            x1, y1, x2, y2 = rect
            cx = (x1 + x2) // 2  
            cy = (y1 + y2) // 2  

            best_match_id = None
            best_iou = 0.0
            best_distance = float('inf')
            
           
            for id, prev_box in self.previous_boxes.items():
                if self.frames_without_detection.get(id, 0) < 15: 
                    iou = calculate_iou([x1, y1, x2, y2], prev_box)
                    if iou > 0.2 and iou > best_iou: 
                        best_match_id = id
                        best_iou = iou
            
            
            if best_match_id is None:
                for id, pt in self.center_points.items():
                    if self.frames_without_detection.get(id, 0) < 12: 
                        dist = math.hypot(cx - pt[0], cy - pt[1])
                        if dist < 45 and dist < best_distance: 
                            best_match_id = id
                            best_distance = dist
            
            if best_match_id is not None:
                
                self.center_points[best_match_id] = (cx, cy)
                self.frames_without_detection[best_match_id] = 0
                self.previous_boxes[best_match_id] = [x1, y1, x2, y2]
                objects_bbs_ids.append([x1, y1, x2, y2, best_match_id])
                matched_objects.add(best_match_id)
            else:
                
                self.center_points[self.id_count] = (cx, cy)
                self.frames_without_detection[self.id_count] = 0
                self.previous_boxes[self.id_count] = [x1, y1, x2, y2]
                objects_bbs_ids.append([x1, y1, x2, y2, self.id_count])
                self.id_count += 1

        
        new_center_points = {}
        new_frames_without_detection = {}
        new_previous_boxes = {}
        
        for obj_bb_id in objects_bbs_ids:
            _, _, _, _, object_id = obj_bb_id
            if object_id in self.center_points:
                new_center_points[object_id] = self.center_points[object_id]
                new_frames_without_detection[object_id] = self.frames_without_detection.get(object_id, 0)
                new_previous_boxes[object_id] = self.previous_boxes.get(object_id, [])

        self.center_points = new_center_points.copy()
        self.frames_without_detection = new_frames_without_detection.copy()
        self.previous_boxes = new_previous_boxes.copy()
        
        return objects_bbs_ids
