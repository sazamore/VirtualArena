
using UnityEngine;
using System;
using System.Collections; 
using System.Collections.Generic; 
using System.Net; 
using System.Net.Sockets; 
using System.Text; 
using System.Threading; 

public class Tcp3 : MonoBehaviour {  	
	#region private members 	
	/// <summary> 	
	/// TCPListener to listen for incoming TCP connection 	
	/// requests. 	
	/// </summary> 	
	private TcpListener tcpListener; 
	/// <summary> 
	/// Background thread for TcpServer workload. 	
	/// </summary> 	
	private Thread tcpListenerThread;  	
	/// <summary> 	
	/// Create handle to connected tcp client. 	
	/// </summary> 	
	private TcpClient connectedTcpClient; 	
	#endregion 	
	
	IPAddress localAdd;
	Vector3 pos = Vector3.zero;
	Vector3 pos2 = Vector3.zero;
	private float offset = 40f;

	// Use this for initialization
	void Start () { 		
		// Make the game run as fast as possible
		Application.targetFrameRate = 120;

		// Start TcpServer background thread 		
		tcpListenerThread = new Thread (new ThreadStart(ListenForIncomingRequests)); 		
		tcpListenerThread.IsBackground = true; 		
		tcpListenerThread.Start(); 	
		pos = transform.position;
		pos2 = transform.position;
	}  	

	/// <summary> 	
	/// Runs in background TcpServerThread; Handles incoming TcpClient requests 	
	/// </summary> 	
	private void ListenForIncomingRequests () { 		
		//try { 			
			// Create listener on localhost. 		
		localAdd = IPAddress.Parse ("127.0.0.1");
		tcpListener = new TcpListener(IPAddress.Any, 25001); 			
		tcpListener.Start();              
		Debug.Log("Server is listening");  
		Byte[] bytes = new Byte[1024];  	
		int point = 1;
		while (true) { 				
			using (connectedTcpClient = tcpListener.AcceptTcpClient()) { 					
				// Get a stream object for reading 					
				using (NetworkStream stream = connectedTcpClient.GetStream()) { 						
					int length; 						
					// Read incoming stream into byte arrary. 						
					while ((length = stream.Read(bytes, 0, bytes.Length)) != 0) { 							
						var incomingData = new byte[length]; 							
						Array.Copy(bytes, 0, incomingData, 0, length); 

						// Convert byte array to string message. 							
						string clientMessage = Encoding.ASCII.GetString(incomingData);
						//float[] floatData = Array.ConvertAll(clientMessage.Split(','),float.Parse);

						//Debug.Log("client message received as: " + clientMessage);

						if (point ==1){

							//Convert to vector 3 for position update
							pos = 2f*StringToVector3(clientMessage);
							point = 2;
							Debug.Log (pos);
						}
						else if (point ==2){
							pos2 = 2f*StringToVector3 (clientMessage);
							point = 1;
							Debug.Log (pos2);
						}
						} 					
					} 				
				}
			}
		} 		
		//} 


	// TODO: fix this biz! Update is called once per frame
	void FixedUpdate () { 		
		//	if (Input.GetKeyDown(KeyCode.Space)) {             
		//		SendMessage();

		//Get angle between the two points (assume pos1 is forwardmost)
		Vector3 targetDir = pos - pos2;
		Vector3 newDir = Vector3.RotateTowards (transform.forward, targetDir, 1.1f * Time.deltaTime, 0.0f);


		//TODO: angle offset - the axes have to be offset by the Leap position (about 40 deg in X (pitch) position.
		//TODO: send a message to the server, and then have python start the script automatically upon receipt.

		Camera.main.gameObject.transform.position = Vector3.MoveTowards (transform.position, pos, 10f * Time.deltaTime);
		//Camera.main.gameObject.transform.rotation = Quaternion.LookRotation (newDir);//*(Quaternion.AngleAxis(offset, Vector3.forward));
		//Camera.main.gameObject.transfo	rm.rotation = Quaternion.RotateTowards(transform.rotation, pos, Time.deltaTime);
		Camera.main.gameObject.transform.rotation = Quaternion.FromToRotation(pos, pos2);

		//Camera.main.gameObject.transform.position = Vector3.Lerp (this.transform.position, pos);/pos;
	//	} 	
	}  	
	  	
	/// <summary> 	
	/// Send message to client using socket connection. 	
	/// </summary> 	
	private void SendMessage() { 		
		if (connectedTcpClient == null) {             
			return;         
		}  		
		
		try { 			
			// Get a stream object for writing. 			
			NetworkStream stream = connectedTcpClient.GetStream(); 			
			if (stream.CanWrite) {                 
				string serverMessage = "This is a message from your server."; 			
				// Convert string message to byte array.                 
				byte[] serverMessageAsByteArray = Encoding.ASCII.GetBytes(serverMessage); 				
				// Write byte array to socketConnection stream.               
				stream.Write(serverMessageAsByteArray, 0, serverMessageAsByteArray.Length);               
				//Debug.Log("Server sent his message - should be received by client");           
			}       
		} 		
		catch (SocketException socketException) {             
			Debug.Log("Socket exception: " + socketException);         
		} 	
	}

	public static Vector3 StringToVector3(string sVector)
	{
		// Remove the parentheses
		if (sVector.StartsWith("(") && sVector.EndsWith(")"))
		{
			sVector = sVector.Substring(1, sVector.Length - 2);
		}
		
		// split the items
		string[] sArray = sVector.Split(',');
		
		// store as a Vector3
		Vector3 result = new Vector3(
			float.Parse(sArray[0]),
			float.Parse(sArray[1]),
			float.Parse(sArray[2]));
		Debug.Log (result);
		return result;
	}
}