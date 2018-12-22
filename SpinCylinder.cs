using System.Collections;
using UnityEngine;

public class SpinCylinder : MonoBehaviour {
	//Controller for rotating the cylinder ("drum") at some speed. Currently set CW rotation.

	public float speed = 1.0f;
	void Start()
	{
		// Make the game run as fast as possible
		Application.targetFrameRate = 120;
	}
	
	// Update is called once per frame
	void Update (){
	
		if (Input.GetKey (KeyCode.RightArrow)) {
			speed = speed+1.0f;	//turn view to right
		}
		if (Input.GetKey (KeyCode.LeftArrow)) {
			speed = speed-1.0f; //turn view to left
		}

		transform.Rotate(Vector3.forward, speed * Time.deltaTime);

	}
}
