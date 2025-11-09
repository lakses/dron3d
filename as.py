from pioneer_sdk import Pioneer, Camera
import cv2
import numpy as np
import time
import torch
import torchvision.transforms as transforms

if __name__ == "__main__":
    print(
        """
    1 -- arm
    2 -- disarm
    3 -- takeoff
    4 -- land
    ↶q  w↑  e↷    i-↑
    ←a      d→     k-↓
        s↓"""
    )
    pioneer_mini = Pioneer()
    camera = Camera()
    min_v = 1300
    max_v = 1700
    
    # Загрузка модели MiDaS
    print("Loading MiDaS model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
    model.to(device)
    model.eval()
    
    # Трансформации для MiDaS
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    transform = midas_transforms.small_transform
    
    print("MiDaS model loaded successfully!")
    
    try:
        while True:
            ch_1 = 1500
            ch_2 = 1500
            ch_3 = 1500
            ch_4 = 1500
            ch_5 = 2000
            frame = camera.get_frame()
            
            if frame is not None:
                camera_frame = cv2.imdecode(
                    np.frombuffer(frame, dtype=np.uint8), cv2.IMREAD_COLOR
                )
                
                # Обработка кадра через MiDaS
                img_rgb = cv2.cvtColor(camera_frame, cv2.COLOR_BGR2RGB)
                input_batch = transform(img_rgb).to(device)
                
                with torch.no_grad():
                    prediction = model(input_batch)
                    
                    # Интерполяция до исходного размера
                    prediction = torch.nn.functional.interpolate(
                        prediction.unsqueeze(1),
                        size=camera_frame.shape[:2],
                        mode="bicubic",
                        align_corners=False,
                    ).squeeze()
                
                depth_map = prediction.cpu().numpy()
                
                # Нормализация для отображения
                depth_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
                depth_colored = cv2.applyColorMap(np.uint8(depth_normalized), cv2.COLORMAP_JET)
                
                # Показ оригинального кадра и карты глубины
                cv2.imshow("pioneer_camera_stream", camera_frame)
                cv2.imshow("MiDaS Depth Map", depth_colored)
                
                # Вывод информации о глубине
                print(f"Depth map - Min: {depth_map.min():.3f}, Max: {depth_map.max():.3f}, Mean: {depth_map.mean():.3f}")
            
            key = cv2.waitKey(1)
            if key == 27:  # esc
                print("esc pressed")
                cv2.destroyAllWindows()
                pioneer_mini.land()
                break
            elif key == ord("1"):
                pioneer_mini.arm()
            elif key == ord("2"):
                pioneer_mini.disarm()
            elif key == ord("3"):
                time.sleep(2)
                pioneer_mini.arm()
                time.sleep(1)
                pioneer_mini.takeoff()
                time.sleep(2)
            elif key == ord("4"):
                time.sleep(2)
                pioneer_mini.land()
                time.sleep(2)
            elif key == ord("w"):
                ch_3 = min_v
            elif key == ord("s"):
                ch_3 = max_v
            elif key == ord("a"):
                ch_4 = min_v
            elif key == ord("d"):
                ch_4 = max_v
            elif key == ord("q"):
                ch_2 = 2000
            elif key == ord("e"):
                ch_2 = 1000
            elif key == ord("i"):
                ch_1 = 2000
            elif key == ord("k"):
                ch_1 = 1000
            
            pioneer_mini.send_rc_channels(
                channel_1=ch_1,
                channel_2=ch_2,
                channel_3=ch_3,
                channel_4=ch_4,
                channel_5=ch_5,
            )
            time.sleep(0.02)
    finally:
        time.sleep(1)
        pioneer_mini.land()
        pioneer_mini.close_connection()
        del pioneer_mini
        cv2.destroyAllWindows()