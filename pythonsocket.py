import sys
import socket
import json

host = 'localhost' 
port = 50000
backlog = 5 
size = 1024 

address = ('localhost',25001)

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM) 
s.connect(address) 
#s.listen(backlog) 

position = sys.argv[1] #"0,1,0"
pos2 = sys.argv[2]
print position

s.sendall(position,position2)

#while 1:
 #   client, address = s.accept() 
 #   print "Client connected."
 #   while 1:
 #       data = client.recv(size)
 #       if data == "ping":
 #           print ("Unity Sent: " + str(data))
 #           client.send("pong")
 #       else:
 #           client.send("Bye!")
 #           print ("Unity Sent Something Else: " + str(data))
 #           client.close()
 #           break
