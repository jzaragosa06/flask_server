Changes:
in the results of forecast, seasonality, and trend
operation, we will be using the original column name in describing the column name of the results Dataframe.
Effect:
In the web application, we shall update the way we access the json data. that is we shall use more bracket operator than the dot operator.



Reminder:
1. in order for the modules to work/get imported properly, we started the import from the app, since it contains **init**.py file, this will be treated as a module.

2. We will be updating the forecast feature. We will only use without-refit. That is, 
we will use refit = False. This decision is to make the code logic simple as well as the selection 
of the desired resul. 
    A. In requesting for forecast result from the server, retain the 'without-refit' parameter. 