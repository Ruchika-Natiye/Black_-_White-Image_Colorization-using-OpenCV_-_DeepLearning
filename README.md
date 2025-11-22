# Black_-_White-Image_Colorization-using-OpenCV_-_DeepLearning

## 📸 Black & White Image Colorization using OpenCV & Deep Learning
Transform old black-and-white images into vibrant, realistic color using a pretrained deep learning model.This project uses the OpenCV DNN module with a Caffe-based colorization model to automatically add colors to grayscale images.

## 🚀 Features
### ✔ Deep Learning Based Colorization
      1.Uses the official colorization_release_v2.caffemodel
      2.Adds color based on 313 ab cluster centers
      3.Produces natural color tones

### ✔ Simple & Fast
      1.Requires only OpenCV and NumPy
      2.Works on CPU—no GPU needed
      3.Output generated in milliseconds

### ✔ Side-by-Side Comparison
      1.original grayscale and colorized results together.

## 📂 Project Structure
![Project Structure](https://github.com/Ruchika-Natiye/Black_-_White-Image_Colorization-using-OpenCV_-_DeepLearning/blob/14e2b0dd58fe1220647762b5b9b43f21a18bf1af/project%20structure.png)

## Python Script
```python
import numpy as np
import cv2
from cv2 import dnn
#Model file paths----#
proto_file=r"C:\Users\Dell\OneDrive\Desktop\python\black_n_white_imagecolorization_using_opencv_and_deeplearning\Model\colorization_deploy_v2.prototxt"
model_file=r"C:\Users\Dell\OneDrive\Desktop\python\black_n_white_imagecolorization_using_opencv_and_deeplearning\Model\colorization_release_v2.caffemodel"
hull_pts=r"C:\Users\Dell\OneDrive\Desktop\python\black_n_white_imagecolorization_using_opencv_and_deeplearning\Model\pts_in_hull.npy"
img_path=r"C:\Users\Dell\OneDrive\Desktop\python\black_n_white_imagecolorization_using_opencv_and_deeplearning\scenery.png"
#Reading the model params----#
net=dnn.readNetFromCaffe(proto_file,model_file)
kernel=np.load(hull_pts)
#Reading and preprocessing image#
img=cv2.imread(img_path)
scaled=img.astype("float32")/255.0
lab_img=cv2.cvtColor(scaled,cv2.COLOR_BGR2LAB)
# add the cluster centers as 1x1 convolutions to the model
class8 = net.getLayerId("class8_ab")
conv8 = net.getLayerId("conv8_313_rh")
pts = kernel.transpose().reshape(2, 313, 1, 1)
net.getLayer(class8).blobs = [pts.astype("float32")]
net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]
# we'll resize the image for the network
resized = cv2.resize(lab_img, (224, 224))
# split the L channel
L = cv2.split(resized)[0]
# mean subtraction
L -= 50
# predicting the ab channels from the input L channel
net.setInput(cv2.dnn.blobFromImage(L))
ab_channel = net.forward()[0, :, :, :].transpose((1, 2, 0))
# resize the predicted 'ab' volume to the same dimensions as our
# input image
ab_channel = cv2.resize(ab_channel, (img.shape[1], img.shape[0]))
# Take the L channel from the image
L = cv2.split(lab_img)[0]
# Join the L channel with predicted ab channel
colorized = np.concatenate((L[:, :, np.newaxis], ab_channel), axis=2)
# Then convert the image from Lab to BGR 
colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)
colorized = np.clip(colorized, 0, 1)
# change the image to 0-255 range and convert it from float32 to int
colorized = (255 * colorized).astype("uint8")
# Let's resize the images and show them together
img = cv2.resize(img,(640,640))
colorized = cv2.resize(colorized,(640,640))
result = cv2.hconcat([img,colorized])
cv2.imshow("Grayscale -> Colour", result)
cv2.waitKey(0)
```
## Working
### 1) Imports
```python
import numpy as np
import cv2
from cv2 import dnn
```
 * numpy for numeric arrays and reshaping.
 
 * cv2 is OpenCV (image I/O, color conversions, resizing, display).
 
 * cv2.dnn provides the deep learning (Caffe) model loading and inference API.

### 2) File paths / constants
```python
proto_file = r"Model/colorization_deploy_v2.prototxt"
model_file = r"Model/colorization_release_v2.caffemodel"
hull_pts = r"Model/pts_in_hull.npy"
img_path = "scenery.png"
```
*   proto_file: network architecture (Caffe .prototxt).

*   model_file: trained weights (.caffemodel).

*   hull_pts: pts_in_hull.npy contains the 313 cluster centers in ab color space used by the model.

*   img_path: the input image to colorize.

### 3) Load the model and cluster centers
```python
net = dnn.readNetFromCaffe(proto_file, model_file)
kernel = np.load(hull_pts)
```
* readNetFromCaffe loads the Caffe model into an OpenCV DNN Net object.

* kernel will be a numpy array loaded from pts_in_hull.npy. Its expected shape is (313, 2) (313 cluster centers, each with an a and b value).

### 4) Read and prepare the input image
```python
img = cv2.imread(img_path)
scaled = img.astype("float32") / 255.0
lab_img = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)
```
* cv2.imread loads image in BGR uint8 (shape: H x W x 3).

  * If img is None, the path was wrong or image unreadable — add a check.

* scaled converts pixels to floats in range [0.0, 1.0] (type float32) — required for accurate LAB conversion.

* cv2.cvtColor(..., cv2.COLOR_BGR2LAB) converts the float BGR image to LAB color space.

  * LAB channels: L (lightness), a (green–red), b (blue–yellow).

  * After conversion with float inputs in range 0..1, L is approx 0..100 and a,b centered around 0 (but exact ranges depend on implementation).

### 5) Prepare and inject cluster centers into the network
 ```python
class8 = net.getLayerId("class8_ab")
conv8 = net.getLayerId("conv8_313_rh")
pts = kernel.transpose().reshape(2, 313, 1, 1)
net.getLayer(class8).blobs = [pts.astype("float32")]
net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]
 ```
* getLayerId("class8_ab") and getLayerId("conv8_313_rh") retrieve internal layer indices by name.

* pts = kernel.transpose().reshape(2, 313, 1, 1) shapes the cluster centers to shape (2, 313, 1, 1) so they can be set as layer blobs:

   * After transpose, pts[0] is the a values and pts[1] the b values.

* net.getLayer(class8).blobs = [pts.astype("float32")] inserts these cluster centers into the class8_ab layer as its learned “weights”.

* net.getLayer(conv8).blobs = [np.full([1, 313], 2.606,...)] sets a prior scaling or bias (the constant 2.606 is used in the original implementation).

### 6) Resize input for the network and prepare L channel
```python
resized = cv2.resize(lab_img, (224, 224))
L = cv2.split(resized)[0]
L -= 50
```
* The network expects a 224×224 L-channel input (as in the original model).

* resized is LAB image at the network input size.

* L = cv2.split(resized)[0] extracts the L channel (shape 224 x 224).

* L -= 50 subtracts 50 (mean-centering). This normalization mirrors the preprocessing used when the model was trained.

### 7) Forward pass to predict ab channels
```python
net.setInput(cv2.dnn.blobFromImage(L))
ab_channel = net.forward()[0, :, :, :].transpose((1, 2, 0))
```
* cv2.dnn.blobFromImage(L) converts the single-channel L into a 4D blob with shape (1, 1, 224, 224) (batch, channels, height, width) as expected by the DNN.

* net.forward() runs inference and returns the network output. For this model the output shape is roughly (1, 313, 56, 56) or similar — i.e., a distribution over 313 bins for each spatial cell.

* [0, :, :, :] selects the first (and only) batch item.

* .transpose((1, 2, 0)) reorders the axes so the result becomes (H_net, W_net, 313). However, note: in many implementations an extra step maps the 313-d distribution to 2 channels (real a and b) before resizing. In this script the model output already yields ab predictions shaped as 2 channels after internal processing.

### 8) Resize predicted ab to original image size
```python
ab_channel = cv2.resize(ab_channel, (img.shape[1], img.shape[0]))
L_original = cv2.split(lab_img)[0]
```
* cv2.resize upscales the predicted chroma channels to the original image width and height: (W, H) where img.shape[1] is width and img.shape[0] is height.

* L_original extracts the original L channel from lab_img (same size as img), so merging will use the original image’s luminance (preserves fine detail and resolution).

### 9) Merge L + predicted ab, convert back to BGR, clip and convert types
```python
colorized = np.concatenate((L_original[:, :, np.newaxis], ab_channel), axis=2)
colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)
colorized = np.clip(colorized, 0, 1)
colorized = (255 * colorized).astype("uint8")
```
* np.concatenate: forms a LAB image (H x W x 3) by stacking L (shape H x W x 1) with ab_channel (shape H x W x 2).

* cv2.cvtColor(..., cv2.COLOR_LAB2BGR): converts the LAB color image back to BGR color space.

* np.clip(..., 0, 1) ensures values remain in the normalized float range [0,1]. This is important to avoid wraparound when converting to uint8.

* (255 * colorized).astype("uint8") scales floats back to 0-255 and casts to uint8 for display/saving.

### 10) Display the result side-by-side
```python
result = cv2.hconcat([img, colorized])
cv2.imshow("Grayscale → Colour", result)
cv2.waitKey(0)
```
* cv2.hconcat horizontally concatenates the original image and the colorized image for comparison.

* cv2.imshow opens a GUI window with the result.

* cv2.waitKey(0) waits indefinitely for a key press to close the window.
  
## 📊 Output Example
	                                                                                                         
![image alt](https://github.com/Ruchika-Natiye/Black_-_White-Image_Colorization-using-OpenCV_-_DeepLearning/blob/da0670fad62b60e157f03962d60cd2668f56039a/Output1.png)

## 📚 References

* Zhang, Richard, et al. "Colorful Image Colorization." ECCV 2016.

* OpenCV DNN Documentation: https://docs.opencv.org/

## ❤️ Contributing

Pull requests are welcome!
If you find a bug or want a feature, feel free to open an issue.

## ⭐ Show Support

If this project helped you, please ⭐ the repository — it motivates me to create more!
