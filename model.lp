\ Model Pump optimisation
\ LP format - for model browsing. Use MPS format to capture full model detail.
\ Signature: 0x6365c154702fdb36
Maximize
  - Min_distance_to_pump
Subject To
 define_xdiff[0]: - Position[0] + xdiff[0] = 11.43349811
 define_ydiff[0]: - Position[1] + ydiff[0] = -10.98081621
 define_xdiff[1]: - Position[0] + xdiff[1] = 11.44427405
 define_ydiff[1]: - Position[1] + ydiff[1] = -10.99065103
 define_xdiff[2]: - Position[0] + xdiff[2] = 11.42095355
 define_ydiff[2]: - Position[1] + ydiff[2] = -11.00194111
 D**2_<=_dist0**2: [ Min_distance_to_pump ^2 - xdiff[0] ^2 - ydiff[0] ^2 ]
   <= 0
 D**2_<=_dist1**2: [ Min_distance_to_pump ^2 - xdiff[1] ^2 - ydiff[1] ^2 ]
   <= 0
 D**2_<=_dist2**2: [ Min_distance_to_pump ^2 - xdiff[2] ^2 - ydiff[2] ^2 ]
   <= 0
Bounds
 xdiff[0] free
 xdiff[1] free
 xdiff[2] free
 ydiff[0] free
 ydiff[1] free
 ydiff[2] free
End
