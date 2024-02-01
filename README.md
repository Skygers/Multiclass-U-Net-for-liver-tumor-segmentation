# Multiclass U-Net Segmentation for Liver Tumor Detection in CT-Scan Images

This repository dedicated to liver tumor detection in CT-scan images through an advanced multiclass U-Net segmentation approach. Leveraging state-of-the-art techniques such as window leveling, window blending, and one-hot semantic segmentation, the method aims to enhance the accuracy and efficiency of liver tumor identification.

### Key Features:

1. **UNet Architecture:**
   - This repository utilizes the powerful U-Net architecture, a convolutional neural network (CNN) designed for image segmentation. This architecture is particularly effective in handling medical imaging tasks, providing accurate segmentation results. You can see the model in [Multiclass U-Net Model](https://github.com/Skygers/Multiclass-U-Net-for-liver-tumor-segmentation/blob/c89a803eb27ac44022f4e16fc42bd6af42135348/Model%20U-Net/U-Net-Model.py)
   - ![Multiclass U-Net](https://github.com/Skygers/Multiclass-U-Net-for-liver-tumor-segmentation/blob/eeabfbe9f6ac0effda82648f425fb032196144dd/Model%20U-Net/U-Net%20Visualisasi.png)
2. **Multiclass Semantic Segmentation:**
   - Unlike traditional binary segmentation, our approach supports multiclass segmentation. This means the model can distinguish between different classes of tissues, allowing for more nuanced and detailed segmentation, crucial for accurate liver tumor detection.
   - ![Multiclass Semantic Segmentation](https://github.com/Skygers/Multiclass-U-Net-for-liver-tumor-segmentation/blob/eeabfbe9f6ac0effda82648f425fb032196144dd/Gambar/mask%20liver%20tiff.jpg)

3. **CT-Scan Image Processing:**
   - Implement window leveling and window blending techniques to preprocess CT-scan images. These methods enhance the visibility of structures, making it easier for the model to identify and delineate liver tumors against the complex background of CT images.
   - **Result Window Leveling and Window Blending Method**

     ![Multiclass Semantic Segmentation](https://github.com/Skygers/Multiclass-U-Net-for-liver-tumor-segmentation/blob/c89a803eb27ac44022f4e16fc42bd6af42135348/Gambar/volume-1_slice_61.jpg)

     This method reference
     [window blending](https://sv-journal.org/2019-5/06/)
4. **One-Hot Semantic Segmentation (OHESS):**
   - this repository employs one-hot encoding for semantic segmentation. This technique enables the model to assign each pixel to a specific class, facilitating precise identification of liver tumors and surrounding tissues.
   - **Result OHESS Liver Class**

     ![Multiclass Semantic Segmentation](https://github.com/Skygers/Multiclass-U-Net-for-liver-tumor-segmentation/blob/c89a803eb27ac44022f4e16fc42bd6af42135348/Gambar/one%20hot%20liver.jpg)
   - **Result OHESS Tumor Class**

     ![Multiclass Semantic Segmentation](https://github.com/Skygers/Multiclass-U-Net-for-liver-tumor-segmentation/blob/c89a803eb27ac44022f4e16fc42bd6af42135348/Gambar/one%20hot%20mask.jpg)
### How to Use:

1. **Dataset Preparation:**
   - Organize your CT-scan dataset, ensuring proper labeling for liver tumor regions and background. This is crucial for training the model effectively.
   - https://www.kaggle.com/datasets/andrewmvd/liver-tumor-segmentation

2. **Model Training:**
   - Utilize the provided training scripts to train the UNet model on your dataset. Tweak hyperparameters as needed for optimal performance.
   - you can see in this [Model Train](https://github.com/Skygers/Multiclass-U-Net-for-liver-tumor-segmentation/blob/c89a803eb27ac44022f4e16fc42bd6af42135348/u-net-train-multiclass-semantic-liver-tumor.ipynb)

3. **Inference and Evaluation:**
   - Run the inference scripts on new CT-scan images to detect liver tumors. Evaluate the model's performance using metrics like Dice Coefficient Similarity and Intersection Over Union for multiclass.

4. **Customization:**
   - Feel free to customize the model architecture, training pipeline, or post-processing steps to better suit your specific requirements or dataset characteristics.
