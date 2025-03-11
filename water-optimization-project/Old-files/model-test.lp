\ LP format - for model browsing. Use MPS format to capture full model detail.
\ Signature: 0xda3e09db2d173b9c
Minimize
  32 u[0] + 30 u[1] + 40 u[2]
Subject To
 define_xdiff[0]: x + xdiff[0] = 358
 define_ydiff[0]: y + ydiff[0] = 309
 define_xdiff[1]: x + xdiff[1] = 275
 define_ydiff[1]: y + ydiff[1] = 126
 define_xdiff[2]: x + xdiff[2] = 424
 define_ydiff[2]: y + ydiff[2] = 154
 define_u[0]: [ u[0] ^2 - xdiff[0] ^2 - ydiff[0] ^2 ] >= 0
 define_u[1]: [ u[1] ^2 - xdiff[1] ^2 - ydiff[1] ^2 ] >= 0
 define_u[2]: [ u[2] ^2 - xdiff[2] ^2 - ydiff[2] ^2 ] >= 0
Bounds
 xdiff[0] free
 xdiff[1] free
 xdiff[2] free
 ydiff[0] free
 ydiff[1] free
 ydiff[2] free
End
