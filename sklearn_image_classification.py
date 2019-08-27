import cv2
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from collections import Counter
from skimage.feature import local_binary_pattern, greycomatrix, greycoprops
from skimage.filters import gabor
from pandas_ml import ConfusionMatrix
from sklearn.metrics import classification_report, accuracy_score
from PIL import Image

sift_feature_size = 10 * 128
hog_feature_size = 10 * 128


def get_sift_features(algorithm, img, feature_size):
    kp, des = algorithm.detectAndCompute(img, None)
    if des is not None:
        flattened_descriptor = des.flatten()
        if flattened_descriptor.size < feature_size:
            sift_features = np.concatenate(
                [flattened_descriptor, np.zeros(feature_size - flattened_descriptor.size)])
        else:
            sift_features = flattened_descriptor[:feature_size]
    else:
        sift_features = np.zeros(feature_size)
    return sift_features


def get_hog_features(img, feature_size):
    hog = cv2.HOGDescriptor()
    hog_des = hog.compute(img)
    if hog_des is not None:
        flattened_descriptor = hog_des.flatten()
        if flattened_descriptor.size < feature_size:
            hog_features = np.concatenate(
                [flattened_descriptor, np.zeros(feature_size - flattened_descriptor.size)])
        else:
            hog_features = flattened_descriptor[:feature_size]
    else:
        hog_features = np.zeros(feature_size)
    return hog_features


def get_color_based_features(image):
    channels = cv2.split(image)
    colors = ("b", "g", "r")
    features = []
    # loop over the image channels
    for (chan, color) in zip(channels, colors):
        # create a histogram for the current channel and concatenate the resulting histograms for each channel
        hist = cv2.calcHist([chan], [0], None, [256], [0, 256])
        features.extend(hist)
    return np.array(features).flatten()


def extract_basic_image_features(image_path):
    img = Image.open(image_path)  # One image at a time
    img_gray = img.convert('L')  # Converting to gray scale
    img_arr = np.array(img_gray.getdata()).reshape(img.size[1], img.size[0])  # Converting to array
    # LBP
    feat_lbp = local_binary_pattern(img_arr, 5, 2, 'uniform').reshape(img.size[0] * img.size[1])
    lbp_hist, _ = np.histogram(feat_lbp, 8)
    lbp_hist = np.array(lbp_hist, dtype=float)
    lbp_prob = np.divide(lbp_hist, np.sum(lbp_hist))
    lbp_energy = np.nansum(lbp_prob ** 2)
    lbp_entropy = -np.nansum(np.multiply(lbp_prob, np.log2(lbp_prob)))
    # GLCM
    grey_co_mat = greycomatrix(img_arr, [2], [0], 256, symmetric=True, normed=True)
    contrast = greycoprops(grey_co_mat, prop='contrast')
    dissimilarity = greycoprops(grey_co_mat, prop='dissimilarity')
    homogeneity = greycoprops(grey_co_mat, prop='homogeneity')
    energy = greycoprops(grey_co_mat, prop='energy')
    correlation = greycoprops(grey_co_mat, prop='correlation')
    feat_glcm = np.array([contrast[0][0], dissimilarity[0][0], homogeneity[0][0], energy[0][0], correlation[0][0]])
    # Gabor filter
    gabor_filt_real, gaborFilt_imag = gabor(img_arr, frequency=0.6)
    gaborFilt = (gabor_filt_real ** 2 + gaborFilt_imag ** 2) // 2
    gabor_hist, _ = np.histogram(gaborFilt, 8)
    gabor_hist = np.array(gabor_hist, dtype=float)
    gabor_prob = np.divide(gabor_hist, np.sum(gabor_hist))
    gabor_energy = np.nansum(gabor_prob ** 2)
    gabor_entropy = -np.nansum(np.multiply(gabor_prob, np.log2(gabor_prob)))
    # Concatenating features(2+5+2)
    concat_feat = np.concatenate(([lbp_energy, lbp_entropy], feat_glcm, [gabor_energy, gabor_entropy]), axis=0)
    trainLabel = np.array(concat_feat)  # Conversion from list to array
    return trainLabel


# Feature extractor
def extract_image_features(image_path):
    try:
        sift = cv2.xfeatures2d.SIFT_create()
        image = cv2.imread(image_path)
        original_image = cv2.imread(image_path)
        # original image
        original_image = cv2.resize(original_image, (64, 128))
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        original_image = cv2.filter2D(original_image, -1, kernel)
        original_image = cv2.filter2D(original_image, -1, kernel)
        # Convert to gray
        img = cv2.resize(image, (120, 128))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        img = cv2.filter2D(img, -1, kernel)
        img = cv2.filter2D(img, -1, kernel)
        # get color features
        color_based_features = get_color_based_features(original_image)
        # get basic features
        basic_image_features = extract_basic_image_features(image_path)
        # sift features
        sift_features = get_sift_features(sift, img, sift_feature_size)
        # hog features
        hog_features = get_hog_features(original_image, hog_feature_size)

        return color_based_features, basic_image_features, sift_features, hog_features
    except cv2.error as e:
        print('Error: ', e)
        return None


def extract_features_all_images(images_path):
    print("extracting images features..")
    files = [os.path.join(images_path, p) for p in sorted(os.listdir(images_path))]
    image_count = 0
    color_based_features, basic_image_features, sift_features, hog_features = [], [], [], []
    for f in files:
        name = f.split('/')[-1].lower()
        image_count = image_count + 1
        if image_count % 1000 == 0:
            print(name, image_count)
        color_based_feature, basic_image_feature, sift_feature, hog_feature = extract_image_features(f)
        color_based_features.append(color_based_feature)
        basic_image_features.append(basic_image_feature)
        sift_features.append(sift_feature)
        hog_features.append(hog_feature)

    color_based_feature_reduced = perform_feature_reduction(color_based_features)
    basic_image_feature_reduced = perform_feature_reduction(basic_image_features)
    sift_feature_reduced = perform_feature_reduction(sift_features)
    hog_feature_reduced = perform_feature_reduction(hog_features)

    return np.concatenate([color_based_feature_reduced, basic_image_feature_reduced, sift_feature_reduced,
                           hog_feature_reduced], axis=1)


def perform_feature_reduction(image_features_list):
    print("Performing feature reduction..")
    # Standardizing the features
    x = StandardScaler().fit_transform(image_features_list)
    pca = PCA(n_components=min(50, len(image_features_list[0])))
    principal_components = pca.fit_transform(x)
    return principal_components


def load_data(data_path, labels_path):
    with open(labels_path, "r") as ins:
        labels = []
        for line in ins:
            labels.append(line.strip())
    return extract_features_all_images(data_path), np.transpose(np.reshape(labels, (-1, len(labels))))


def train_model(x_train, y_train):
    print("training model..")
    classifier = MLPClassifier(hidden_layer_sizes=(5, 5, 5), max_iter=10000)
    # classifier = RandomForestClassifier()
    classifier.fit(x_train, np.ravel(y_train))
    return classifier


def predict(training_model, x_test):
    print("predicting..")
    return training_model.predict(x_test)


def sample_data(train_data, train_labels):
    print("SMOTE data..")
    sampler = SMOTE(ratio='minority')
    x_sm, y_sm = sampler.fit_resample(train_data, train_labels)
    return x_sm, y_sm


def plot_labels(train_labels):
    train_labels_dict = Counter(train_labels.flatten())
    plt.bar(train_labels_dict.keys(), train_labels_dict.values())
    plt.show()


def classification_metrics(y_test, y_pred):
    if y_test is not None:
        print("printing classification metrics..")
        print("Accuracy: " + str(accuracy_score(y_test, y_pred)))
        print('\n')
        print(classification_report(y_test, y_pred))
        print("F1 Score is ", f1_score(y_test, y_pred, average='weighted'))
        confusion_matrix = ConfusionMatrix(np.ravel(y_test), y_pred)
        print("Confusion matrix:\n%s" % confusion_matrix)
    print("saving to output file..")
    np.savetxt('predictions.txt', y_pred, delimiter=',', fmt="%s")


def main():
    # Train dataset
    split_train_data_mode = 0
    x_train, y_train, x_test, y_test = None, None, None, None
    if split_train_data_mode == 1:
        base_path = "./dataset/traffic-small"
        train_data_set_path = base_path + "/train/"
        train_labels_path = base_path + "/train.labels"
        train_data, train_labels = load_data(train_data_set_path, train_labels_path)
        x_train, x_test, y_train, y_test = train_test_split(train_data, train_labels, test_size=0.15)
        x_train, y_train = sample_data(x_train, y_train)
    else:
        base_path = "./dataset/traffic"
        train_data_set_path = base_path + "/train/"
        train_labels_path = base_path + "/train.labels"
        test_data_set_path = base_path + "/test"
        x_train, y_train = load_data(train_data_set_path, train_labels_path)
        x_train, y_train = sample_data(x_train, y_train)
        x_test = extract_features_all_images(test_data_set_path)

    training_model = train_model(x_train, y_train)
    predictions = predict(training_model, x_test)
    classification_metrics(y_test, predictions)


if __name__ == "__main__":
    main()
