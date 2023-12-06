# Checkout-Store-Assignment

## First part - Python

Had successfully incorporated all the specified modules outlined in the assignment, as detailed in the provided [![link](link)](https://github.com/IordachescuAnca/Checkout-Store-Assignment/tree/main/Checkout).

The file that contains the collective data from weight sensors: [![link](link)](https://github.com/IordachescuAnca/Checkout-Store-Assignment/blob/main/Checkout/data/weights.txt).

The file that contains the collective data from stabilization intervals: [![link](link)](https://github.com/IordachescuAnca/Checkout-Store-Assignment/blob/main/Checkout/data/stab_interval.txt).


## 1 - Client
To run the client file, run the following command from the root directory of the project:

        Usage: python3 -m client [OPTIONS]

        Options:
        --weight-sensor-path TEXT       Path to the file that contains the weight of the sensors.
                                        
        --stable-intervals-path TEXT    Path to the file that contains the values of the stable intervals.

        --server-url TEXT    The server url the client sends a post request to.

        ---weight-treshold INT The value of the weight threshold for checking for any change.



        **Example**: python3 -m client --weight-sensor-path /home/anca/Desktop/Checkout/data/weights.txt --stable-intervals-path /home/anca/Desktop/Checkout/data/stab_interval.txt  --server-url http://192.168.1.233:5000/determine_change/ ---weight-treshold 10


Output:

![Example](https://github.com/IordachescuAnca/Checkout-Store-Assignment/blob/main/imgs/client.png)




## 2 - Server
To run the client file, run the following command from the root directory of the project:

        Usage: python3 server.py


Output:

![Example](https://github.com/IordachescuAnca/Checkout-Store-Assignment/blob/main/imgs/server.png)



## Second part - Computer Vision

## 1 - Multiple object tracking system

To achieve the tracking of multiple objects in a video, two essential components are required: an object detector (example: YOLO) and tracking algorithm (example: DeepSort)

The object detector helps us as it provides initial object locations, can handle occlusions and enables adaptation to different environments by identifying or changing new objects. The choice of using YOLO has been made due to the fact that it is more efficient and enables real-time object detection as the entire image in one forward pass (contrast to Faster-RCNN). Although the pretrained version already achieves high accuracy, fine-tuning is imperative to further enhance the model's performance. In this [![code](link)](https://github.com/IordachescuAnca/Artificial-Intelligence-FMI/blob/main/1st%20year/CV/Bowling/task2_training.ipynb), I specifically fine-tuned a YOLOv5 detector to optimize its ability to detect bowling balls.


The most common metrics are:
- Intersection over Union as it computes the overlap between the ground truth bounding box and the predicted bounding box. Maximizing the intersection is essential to achieve superior performance.
- Mean Average Precision as it computes the overall  precision-recall trade-off.


While YOLO provides bounding box predictions in each individual frame, the tracking algorithm helps by maintain consistent object detection across frames when they are momentarily not visible in certain frames. DeepSort is a popular choice as 
it is based on deep learning approaches (unlike SORT) to extract features from bounding boxes enhances tracking precision when it comes to complex scenarios.


Explaining the object detection system flow:

- Run the YOLO model on each frame of the video to identify and obtain the position of the objects.
- Extract the features from each bounding box using deepsort (a convolutional nn) in order to maintain motion prediction and link object detections across frames -> there might be frames where objects go out of view
- Filter detections that provide low-confidence accuracy or based on object classes

## 2 - Person Reidentification System

The reidentification problem occurs when we need to figure out if pedestrians seen in various camera shots or different clips from the same camera are actually the same people.

To achieve this, we can employ an Object Detector, like the previously mentioned YOLO. This detector helps identify all the bounding boxes in frames captured by the two cameras to locate pedestrians. Subsequently, by examining these bounding boxes, we can determine whether a person is present in two distinct boxes from the two cameras.


To accomplish this, I developed a neural network structured upon a ResNet architecture. This network is designed to learn embeddings for each photo, taking into account that the cosine similarity between two bounding boxes containing the same individual should be maximized using the ProxyAnchorLoss. The dataset used was ![Market 1501](https://zheng-lab.cecs.anu.edu.au/Project/project_reid.html) that contains photos of people  from a total of six cameras were used.


The foundational structure of the architecture underwent testing by utilizing pre-trained ResNet-18 and ResNet-50 models.

The written code: [![link](link)](https://github.com/IordachescuAnca/Checkout-Store-Assignment/tree/main/person_reid).

To run the client file, run the following command from the root directory of the project:

        Usage: python3 -m train [OPTIONS]

        Options:
        --experiment-name TEXT       The name of the experiment (used for the TensorBoard)
                                        
        --dataset-path TEXT    Path to the Market 101 dataset

        --model-dir TEXT    The output directory that saves the best ckpts

        --model-name TEXT    The name of the model

        --batch-size INT     Batch size Values

        --num-workers INT     Number of workers

        --epochs INT     Number of epoch

        --embedding-shape INT    The size of the embedding shape

        --lr     The learning rate value



        **Example**: python3 -m train --experiment-name resnet_18_exo --dataset-path ComputerVision/data/reid/Market101 --model-name resnet18 --batch-size 32



## Tensorboard experiments (exp3- Resnet-18 and exp4- Resnet-50)

![Example](https://github.com/IordachescuAnca/Checkout-Store-Assignment/blob/main/imgs/exp.png)


The metrics employed in this evaluation are Recall@k, a widely-used measure for retrieval tasks that emphasizes the percentage of relevant items successfully retrieved within the top k results. The calculations were performed for k values of 1, 3, 5, 7, and 10.

An alternative strategy could involve training the network using Triplet loss, wherein anchor-positive and anchor-negative pairs are defined. Anchor positives would encompass relations containing photos of the same person, while anchor-negatives would include relations between two different photos. These pairs could be computed at the outset or through the utilization of a sampler.


Now, with the obtained embeddings, the objective is as follows: using the bounding boxes detected by the object detector (OD), we will extract embeddings from each person's photo using the REID model. The unique embeddings, each representing an individual, will be stored in a database. When a new bounding box containing a person is detected, we check if there is an embedding in the database with a cosine similarity greater than a specified threshold. If this condition is met, it indicates that the detected person is someone we have already encountered on the camera, and there is no need to register them as a new unique person in the dataset.
