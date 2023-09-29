
# Demo for daily construction report 

We provide the step-by-step instruction to build the same app as us. Basically, you need to build the backend to handle the model inference and the frontend to allow users to upload images and include more features (e.g., text summarization). By following the instruction, you should be able to have what we demonstrate in the video.

0. To enable the summarization function, you will need to install the bert summarization library through
```
pip install -q bert-extractive-summarizer
```


1. For the backend, please refer to the [official PyTorch tutorial](https://pytorch.org/tutorials/intermediate/flask_rest_api_tutorial.html) for any required installation. In this repo, we simply extend [their repo](https://github.com/avinassh/pytorch-flask-api) to run the model inference and caption summarization. After installing all the dependencies and downloading our files, you should run the backend as follows:
```
FLASK_ENV=development FLASK_APP=demo.py flask run
```
2. For the frontend, you need to first run the server and replace the api in the source code. We develop a simple react web for the interaction (`Node.js` is required). Feel free to extend our work to develop any potential applications with our/your model.
```
cd web_app/
npm install
npm start
```