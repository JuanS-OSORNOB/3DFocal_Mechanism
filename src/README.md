# 3DFocal_Mechanism
This python script serves the purpose of plotting 3D Focal Mechanisms using matplotlib and starting from a database where Focal Mechanism Solutions (FMS) attributes are compiled.

You should have an Excel list or ASCII text file with following information per row:
Latitude (°-coordinate degrees), Longitude (°-coordinate degrees), Depth (km), Strike1 (°-arcdegree), Dip1 (°-arcdegree), Rake1 (°-arcdegree), Strike2 (°-arcdegree), Dip2 (°-arcdegree), Rake2 (°-arcdegree), Moment magnitude (Mw), Scalar Moment (N.m), Moment tensor parameters such as Mnn, Mee, Mdd, Mne, Mnd and Med (N.m [x 1+ePOWER]).

After this you can insert yout FMS list to be read by the python script using pandas dataframe function or any other method of your choice. The beachball list must be organized in this structure*: beachball list=[[R1, C1, NP1],[R2, C2, NP2],...,[Ri, Ci, NPi]] from i = 1 to n number of FMS, where Ri=Mw, Ci=[Xi,Yi,Zi], NPi=[Si, Di, Ri].

*Note: It is not necessary to build the list with both nodal planes (NP) for each beachball as the programm plots the auxiliary plane given that this is always perpendicular to the one inserted by you.

You are set to go, run the script and verify that FMS are plotted with size relative to their magnitude, right polarity and accordingly to their position in space.

Please note that this package is in very early development and is subject to substantial change as we refactor it. 