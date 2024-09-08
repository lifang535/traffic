method = 'after_defending' # 'before_attacking' or 'after_attacking'

config = {
    'input_video_dir': '../input_video',
    
    'output_video_dir': '../output_video', # use
    
    'video_number': 1,
    
    'video_start_id': 4,
    
    'input_image_dir': f'../input_image/{method}', # use
    
    'output_image_dir': f'../output_image/{method}',
    
    'frame_interval': 1, # ms
    
    'monitor_interval': 100, # ms
    
    'qsize_path': f'../picture/{method}/qsize.png',
    'latency_path': f'../picture/{method}/latency.png',
    'times_path': f'../picture/{method}/times.png',
    'flops_path': f'../picture/{method}/flops.png',
    
    'frame_size': (224, 224, 3),
    
    'object_detection': {
        'device': 'cuda:3',
        'yolo-tiny_image_processor_path': '../model/yolo-tiny/yolos-tiny_image_processor.pth',
        'yolo-tiny_model_path': '../model/yolo-tiny/yolos-tiny_model.pth',
    },
    
    'license_recognition': {
        'device': 'cuda:2',
        'easyocr_model_path': '../model/easyocr/easyocr_model.pth',
    },
    
    'person_recognition': {
        'device': 'cpu',
        'face_recognition_model_path': '../model/face_recognition/face_encodings.pkl',
    },
}
