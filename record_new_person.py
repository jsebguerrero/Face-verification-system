def main():
    import cv2
    import os
    from utilities import create_dir, extract_face_image, load_dataset, create_dir
    path = input('ingresar nombre: ')
    path = './faces/' + path
    create_dir(path)
    img_counter = 0
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if img_counter < 200 :  # <---- Check if 5 sec passed
            try:
                frame = extract_face_image(frame)
                img_name = path + '/frame_{}.png'.format(img_counter)
                cv2.imwrite(img_name, frame)
                print("{} written!".format(img_counter))
                img_counter += 1
            except IndexError:
                print('face out of range')
        else:
            print('!done')
            break
    cap.release()
    cv2.destroyAllWindows()

    for realperson in os.listdir('./faces/'):
        if realperson != '5-celebrity-faces-dataset':
            print('working on ', realperson)
            counter = 0
            photos = os.listdir('./faces/' + realperson)
            val = int(len(photos) - 0.6 * len(photos))
            train = len(photos) - val
            print('# train : ', train, '# val : ', val)
            create_dir('./faces/' + realperson + '/train')
            create_dir('./faces/' + realperson + '/val')
            print('created train and test directories')
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
