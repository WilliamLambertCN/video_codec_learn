import cv2
import glob
import os

def video_to_frame(save_path,video_path):
    # save_path : frame save path
    # video_path : source video path
    # max_index :

    # video = cv2.VideoCapture(video_path)
    # index = 0
    # if video.isOpened():
    #     rval,frame = video.read()
    # else:
    #     rval = False
    # while rval:
    #     print(index)
    #     rval,frame = video.read()
    #     print(f"{save_path}/{index:04d}.jpg")
    #     # cv2.imwrite(save_path + '/' + str(index)+'.jpg',frame)
    #     index += 1

    try:
        video = cv2.VideoCapture(video_path)
        index = 1
        assert video.isOpened()
    except:
        print("video not opened")
    else:
        rval,frame = video.read()
        while rval:
            print(f"saved to {save_path}/{index:04d}.jpg")
            cv2.imwrite(f"{save_path}/{index:04d}.jpg",frame)
            index += 1
            rval,frame = video.read()
    finally:
        video.release()

if __name__ == "__main__":
    for i in range(2,4):
        video_path=f"/dataset/TVD/TVD-0{i}.mp4"
        save_path=os.path.splitext(video_path)[0]
        # print(f"mkdir {save_path}")
        os.system(f"mkdir {save_path}")
        video_to_frame(video_path=video_path, save_path=save_path)
        print(f"{video_path} succeed")
