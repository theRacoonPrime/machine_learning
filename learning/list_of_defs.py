import cv2


def parse_txt_annot(img_path, txt_path):
    img = cv2.imread(img_path)
    w = int(img.shape[0])
    h = int(img.shape[1])

    file_label = open(txt_path, "r")
    lines = file_label.read().split('\n')

    boxes = []
    classes = []

    if lines[0] == '':
        return img_path, classes, boxes
    else:
        for i in range(0, int(len(lines))):
            objbud = lines[i].split(' ')
            class_ = int(objbud[0])

            x1 = float(objbud[1])
            y1 = float(objbud[2])
            w1 = float(objbud[3])
            h1 = float(objbud[4])

            xmin = int((x1 * w) - (w1 * w) / 2.0)
            ymin = int((y1 * h) - (h1 * h) / 2.0)
            xmax = int((x1 * w) + (w1 * w) / 2.0)
            ymax = int((y1 * h) + (h1 * h) / 2.0)

            boxes.append([xmin, ymin, xmax, ymax])
            classes.append(class_)

    return img_path, classes, boxes


def countplot_comparison(feature):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 4))
    s1 = sns.countplot(data_df[feature], ax=ax1)
    s1.set_title("Overview Data")

    s2 = sns.countplot(tiff_data[feature], ax=ax2)
    s2.set_title("Tiff Images")

    s3 = sns.countplot(dicom_data[feature], ax=ax3)
    s3.set_title("Dicom Images")

    plt.show()


def normalize(img):
    normalized = cv2.normalize(img, None, alpha=0, beta=200, norm_type=cv2.NORM_MINMAX)
    return normalized
