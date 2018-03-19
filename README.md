# Wrestlebot
Reads from a MongoDB for information about previous matches in order to predict winners of WWE matches

TO DO:
basically all of it for the 3.0 version

1.  update scrape method                                                                                - DONE

2.  define method to get info from MongoDB                                                              - in progress - 95%
3.  define method to build model (linear)                                                               - in progress - 50%

3.1 INTERMEDIATE                                                                                        - not started
    spoof iris categorization model to pull from MongoDB instead of CSV
    intermediate step to output CSV when querying model for wrestler history
    compare resultant inputs to troubleshoot bytes_or_string error during model building

4.  define method to save / load model                                                                  - in progress
5.  define method to query model                                                                        - in progress

6.  define method to build model (deep)                                                                 - not started
7.  define method to build model (wide+deep)                                                            - not started
8.  define method to compared models                                                                    - not started