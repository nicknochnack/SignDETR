# Using VoxDex Transformers for Basic Sign Language Estimation
More of a deep dive into training a DETR model from scratch and all the nuaces with getting object detection running. It was...fun. Anyway, here's a full walkthrough from me to you. Let me know how you go!
“VoxDex is a real-time AI sign language to text system.”

## See it live and in action 📺 - Click the image!
<a href=""><img src="https://i.imgur.com/Om4kU9a.png"/></a>
Link to be added. 

# Setup 🪛
1. Install UV - `pip install uv`
2. Clone the repo - `git clone https://github.com/nicknochnack/SignDETR .`
3. Install all the dependencies `uv sync`

# Collecting images 
1. Update classes in `src/utils/collect_images.py`
2. Run the script `uv run src/utils/collect_images.py`

# Labelling them 
1. Make sure label-studio is installed `uv pip list | grep label-studio`
2. Run the labelling tool `uv run label-studio`
3. Create new project, setup 
4. Labelling shortcuts CTRL + Enter submit, enter number per label 

# Training 🦾
1. Create a checkpoints folder `mkdir checkpoints`
2. Run the training pipeline `uv run src/train.py`

# Running  🚀 
1. To test on your test set, update the checkpoint parameter in `test.py` then run `uv run src/test.py`
2. To run in real time, update the checkpoint parameter in `realtime.py` then run `uv run src/realtime.py`</br> 
<strong>N.B.</strong> you might need need to update your camera parameter in cv2.VideoCapture() to get the right webcam for your machine. 

# Great resources: 
- <a href='https://colab.research.google.com/github/facebookresearch/detr/blob/colab/notebooks/detr_demo.ipynb'>DETR walkthrough</a> - I used this a ton when initially working out how to do this. 


# Who, When, Why?
👨🏾‍💻 Author: Nick Renotte <br />
📅 Version: 1.x<br />
📜 License: This project is licensed under the MIT License </br>
