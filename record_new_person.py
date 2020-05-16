
# --------------------------------------------------------------------------
# --  Record a new person to train the SVM with a new face  ----------------
# --------------------------------------------------------------------------

def main():
    import cv2  # OpenCV library
    import os # Provides functions for interacting with the operating system
    from utilities import create_dir, extract_face_image, load_dataset, create_dir
    
    # Ask for a name for the new person
    path = input('ingresar nombre: ')
    # Create a new subdir for the new person
    path = './faces/' + path
    create_dir(path)
    img_counter = 0
    # Capture video from camera
    cap = cv2.VideoCapture(0)
    
   # --------------------------------------------------------------------------
   # --  Save images containing the face for the new person  ------------------
   # --------------------------------------------------------------------------
    
    while cap.isOpened():
        # Frame-by-frame capture
        ret, frame = cap.read()
        # When no video is returned from de videocamera, the execution ends
        if not ret:
            break
        # When "q" is pressed, the execution ends
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if img_counter < 200 :
            try:
                # Call function to detect and extract the face image from a frame
                frame = extract_face_image(frame)
                # Generate a name depending of the counter
                img_name = path + '/frame_{}.png'.format(img_counter)
                # Save the face as an image
                cv2.imwrite(img_name, frame)
                print("{} written!".format(img_counter))
                img_counter += 1
                
            # Print the next message when a face is not detected
            except IndexError:
                print('face out of range')
                
        # Print the next message and finish this loop when the process ended satisfactorily 
        else:
            print('done!')
            break
        
    # Close all windows
    cap.release()
    cv2.destroyAllWindows()

   # --------------------------------------------------------------------------
   # --  Create train and test directories for a new person  ------------------
   # --------------------------------------------------------------------------

    # Get each new image, verify that it is a new face and create its directories 
    for realperson in os.listdir('./faces/'):
        # Verify if the new image do not correspond to an older image
        if realperson != '5-celebrity-faces-dataset':
            print('working on ', realperson)
            counter = 0
            photos = os.listdir('./faces/' + realperson)
            # Obtain the length of the subdir, it´s necessary to rename each photo  
            val = int(len(photos) - 0.6 * len(photos))
            train = len(photos) - val
            print('# train : ', train, '# val : ', val)
            # Create a directory for trained and other for tested photos 
            create_dir('./faces/' + realperson + '/train')
            create_dir('./faces/' + realperson + '/val')
            print('created train and test directories')
            # Rename photos in the both directories when it's necessary
            for photo in photos:
                if counter != train:
                    os.rename('./faces/' + realperson + '/' + photo, './faces/' + realperson + '/train/' + photo)
                    counter += 1
                else:
                    os.rename('./faces/' + realperson + '/' + photo, './faces/' + realperson + '/val/' + photo)

            print('done for ', realperson)
        else:
            print('finished')


if __name__ == "__main__":
    main()
