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






