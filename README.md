# Googly Eyes

The objective of this service is to receive an image, find the faces and eyes, replace the eyes with googly eyes, and then return the image.

Only `jpeg`, `jpg`, and `png` image formats are allowed.

It used the Haar Cascades classifier with the OpenCV library to find the faces and eyes. The pre-trained Haar Cascades for face and eyes detection are available at `./data`, but it's possible to get in from [OpenCV Github](https://github.com/opencv/opencv/blob/master/data/haarcascades/).

### Image flow

Receive an image, detect faces, get the image face roi, and detect eyes. With the detected eyes, it's possible to draw googly eyes. The pupils are drawn slightly randomized in size and orientation. Then, finally, the image is returned. The pictures are not stored in any part of the service's execution.

### Getting started

The project uses [Poetry](https://python-poetry.org/) version 1.7.1 as a dependency management tool.

Create a development environment:
```
poetry install
```

Run the project:
```
poetry run flask --app app run --debug
```

Invoke the endpoint `/googly-eyes`:
```
curl -X POST -F "image=@<IMAGE_PATH>" http://localhost:5000/googly-eyes --output result.jpeg
```

Replace `<IMAGE_PATH>` with the image path to googlify. A `result.jpeg` image will be saved with the googly eyes.



To run the unit tests:

```
poetry run pytest --cov=app --cov=src tests;
```

Test coverage is currently at 97%

To test with different Haar Cascade pre-trained files, only change the `xml` files path at `config.yaml`.

```
face_cascade_classifier_config: &face_cascade_classifier_config
  path: "./data/haarcascade_frontalface_default.xml"
  scale_factor: 1.3
  min_neighbors: 5


eyes_cascade_classifier_config: &eyes_cascade_classifier_config
  path: "./data/haarcascade_eye.xml"
  scale_factor: 1.2
  min_neighbors: 5
```

### Docker environment

Alternatively, building and running a Docker container with the `googly-eyes` service is possible. This distribution form uses the [gunicorn](https://gunicorn.org/) Python WSGI HTTP server with 2 (two) `sync` workers. It's possible to change this config by changing `gunicorn.conf.py` file:

```
worker_class = "sync"
workers = 2
```


To build the image:
```
docker build -t googly-eyes:latest .
```

Then, to run the container:
```
docker run -p 5000:5000 --name googlify googly-eyes
```

Invoke the endpoint `/googly-eyes`:
```
curl -X POST -F "image=@<IMAGE_PATH>" http://localhost:5000/googly-eyes --output result.jpeg
```

Replace `<IMAGE_PATH>` with the image path to googlify. A `result.jpeg` image will be saved with the googly eyes.


### Future work

* Add new face and eye detection models with better accuracy.