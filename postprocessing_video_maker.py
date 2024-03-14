import cv2
import os
images = []

# field = 'u'
# field = 'v'
field = 'p'

def make_video(field):
    image_folder = "saved\\8_modes\\zeta_0.00\\data\\Ur_5.00\\videos\\frames_{}\\".format(field)
    video_name = 'saved\\8_modes\\zeta_0.00\\data\\Ur_5.00\\videos\\video_{}_reconstruction.avi'.format(field)

    n_frames = 200

    images = [field +"_"+str(i)+".png" for i in range(n_frames)]
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    frame_per_s = n_frames//7

    video = cv2.VideoWriter(video_name, 0,frame_per_s,(width,height))

    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))

    cv2.destroyAllWindows()
    video.release()

make_video('u')
make_video('v')
make_video('p')



