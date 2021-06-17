#!/usr/bin/env python
import rospy
import numpy as np
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import Marker
import matplotlib.pyplot as plt

# roslaunch jackal_gazebo jackal_world.launch config:=front_laser
# roslaunch jackal_viz view_robot.launch

# IMU topic /imu/data
# Odometry topic /odometry/filtered (this sounds like a problem)
# Laser Scan topic /front/scan
# Command velocity topic /cmd_vel
# what's up with /jackal_velocity_controller/cmd_vel and /jackal_velocity_controller/odom

# so basically what I need to do is:
# - subscribe to all the sensors
# - figure out what the system dynamics are, either from your own simple estimate or from digging around the jackal simulator
# - write the Kalman Filter update equations given these dynamics (as well as the sensor dynamics for each sensor, odom and imu should be easy but laser scan might be hard)
# - publish to rviz the current estimate of the robot state (try with different combos of the three sensors)
# - extend the state to include the landmarks

def integrate_dynamics(state, step, output):
	x, y, _, theta, _ = state
	v, omega = output
	return np.array([np.cos(theta) * v * step + x,
					 np.sin(theta) * v * step + y,
					 v,
					 omega * step + theta,
					 omega])

def dynamics(state, step):
	x, y, v, theta, omega = state
	return np.array([np.cos(theta) * v * step + x,
					 np.sin(theta) * v * step + y,
					 v,
					 omega * step + theta,
					 omega]) + process_noise(5)

def linear_dynamics(state, step):
	x, y, v, theta, omega = state
	return np.array([[1, 0, np.cos(theta)*step, -np.sin(theta)*v*step, 0], 
					 [0, 1, np.sin(theta)*step, np.cos(theta)*v*step, 0], 
					 [0, 0, 1, 0, 0], 
					 [0, 0, 0, 1, step], 
					 [0, 0, 0, 0, 1]])

def measurement_model(state): # , landmarks):
	x, y, v, theta, omega = state
	odom_model = np.array([v, omega])
	# if landmarks:
	# 	laser_scan_model = []
	# 	for l in landmarks:
	# 		laser_scan_model.append(np.array([np.sqrt((x - l[0])**2 + (y - l[1])**2),
	# 										  np.arctan2(y - l[1], x -l[0]) - theta]))
	# 	laser_scan_model = np.vstack(laser_scan_model)
	# 	return np.vstack((odom_model, laser_scan_model))
	# else:
	# 	return odom_model
	return odom_model + measurement_noise(2)


def linear_measurement_model(state): # , landmarks):
	x, y, v, theta, omega = state
	odom_model = np.array([[0, 0, 1, 0, 0], 
						   [0, 0, 0, 0, 1]])
	# if landmarks:
	# 	laser_scan_model = []
	# 	for l in landmarks:
	# 		laser_scan_model.append(np.array([[(xhat-l[0])/np.sqrt((xhat-l[0])**2+(yhat-l[1])**2), (yhat-l[1])/np.sqrt((xhat-l[0])**2+(yhat-l[1])**2), 0, 0, 0], 
	# 										  [-(yhat-l[1])/(1+((yhat-l[1])/(xhat-l[0]))**2 * (xhat-l[0])**2), 1/(1+((yhat-l[1])/(xhat-l[0]))**2 * (xhat-l[0])), 0, -1, 0]]))
	# 	laser_scan_model = np.vstack(laser_scan_model)
	# 	return np.vstack((odom_model, laser_scan_model))
	# else:
	# 	return odom_model
	return odom_model

def process_noise(N):
	# what should the covariance be? identity? ones?
	# return np.random.multivariate_normal(np.zeros((N,)), np.identity(5)*0.01)
	return np.random.multivariate_normal(np.zeros((N,)), np.identity(5)*0.00001) #np.array([[0.01825**0.5, 0, 0, 0, 0], [0, 0.08089**0.5, 0, 0, 0], [0, 0, 1e-5, 0, 0], [0, 0, 0, 0.08912**0.5, 0], [0, 0, 0, 0, 0.01]])) # np.ones((N, N))*0.001)#np.identity(N)*0.001)

def measurement_noise(N):
	# what should the covariance be? identity? ones?
	# return np.random.multivariate_normal(np.zeros((N,)), np.identity(2)*0.01)
	return np.random.multivariate_normal(np.zeros((N,)), np.array([[1e-5, 0], [0, 0.01]])) # np.ones((N, N))*0.001)#np.identity(N)*0.001)

# TODO where do I add in the noise vectors??
# TODO I tihnk I need to wrap theta

# 0.0182560775451
# 0.0808943525884
# 0.0218975421455
# 0.0891248631988
# 0.0218975421455

class KalmanFilter:

	def __init__(self, f, h, F, H):
		self.f = f
		self.h = h
		self.F = F
		self.H = H
		self.V = np.identity(5)*0.00001 #np.array([[0.01825**0.5, 0, 0, 0, 0], [0, 0.08089**0.5, 0, 0, 0], [0, 0, 1e-5, 0, 0], [0, 0, 0, 0.08912**0.5, 0], [0, 0, 0, 0, 0.01]]) # np.array([[0.1, 0, 0, 0, 0], [0, 0.1, 0, 0, 0], [0, 0, 1e-5, 0, 0], [0, 0, 0, 0.1, 0], [0, 0, 0, 0, 0.01]]) # np.identity(5)*0.01 # generalize this, what is correct?
		self.W = np.array([[1e-5, 0], [0, 0.01]]) # np.array([[1e-5, 0], [0, 0.01]]) #np.identity(2)*0.01 # generalize this, what is correct?
		self.position = Marker() 
		self.orientation = Marker()
		self.xhat = None
		self.xhat_integrated = None
		self.P = None
		self.initialized1 = False
		self.initialized2 = False
		self.x_history = []
		self.y_history = []
		self.v_history = []
		self.theta_history = []
		self.omega_history = []
		self.x_integrated = []
		self.y_integrated = []
		self.theta_integrated = []
		self.x_actual = []
		self.y_actual = []
		self.theta_actual = []
		self.time = []
		self.time_integrated = []
		self.time_actual = []
		self.plot = True
		# self.odom_sub = rospy.Subscriber("/odometry/filtered", Odometry, self.odom_callback)
		self.odom_sub = rospy.Subscriber("/jackal_velocity_controller/odom", Odometry, self.odom_callback)
		self.filtered_odom_sub = rospy.Subscriber("/odometry/filtered", Odometry, self.filtered_odom_callback)
		self.laser_sub = rospy.Subscriber("/front/scan", LaserScan, self.scan_callback)
		self.position_pub = rospy.Publisher("/rviz/position", Marker, queue_size=100)		
		self.orientation_pub = rospy.Publisher("/rviz/orientation", Marker, queue_size=100)

	def odom_callback(self, data):
		if self.initialized1 and not self.initialized2:
			self.i = 0

			# x = data.pose.pose.position.x
			# y = data.pose.pose.position.y
			# v = data.twist.twist.linear.x
			# q  = data.pose.pose.orientation
			# omega = data.twist.twist.angular.z

			x = self.x
			y = self.y_filtered
			v = self.v
			q = quaternion_from_euler(0, 0, self.theta)
			omega = self.omega

			self.position.type = 3
			self.position.header.frame_id = "odom"
			self.position.header.stamp = rospy.Time.now()
			self.position.pose.position.x = x
			self.position.pose.position.y = y
			self.position.pose.position.z = 0.0
			q0 = quaternion_from_euler(0, 0, 0)
			self.orientation.pose.orientation.x = q0[0]
			self.orientation.pose.orientation.y = q0[1]
			self.orientation.pose.orientation.z = q0[2]
			self.orientation.pose.orientation.w = q0[3]
			self.position.scale.x = 1.0
			self.position.scale.y = 1.0
			self.position.scale.z = 0.1
			self.position.color.a = 1.0
			self.position.color.r = 1.0
			self.position.color.g = 0.0
			self.position.color.b = 0.0

			self.orientation.type = 0
			self.orientation.header.frame_id = "odom"
			self.orientation.header.stamp = rospy.Time.now()
			self.orientation.pose.position.x = x
			self.orientation.pose.position.y = y
			self.orientation.pose.position.z = 0.1
			self.orientation.pose.orientation.x = q[0]#q.x
			self.orientation.pose.orientation.y = q[1]#q.y
			self.orientation.pose.orientation.z = q[2]#q.z
			self.orientation.pose.orientation.w = q[3]#q.w
			self.orientation.scale.x = 1.0
			self.orientation.scale.y = 0.1
			self.orientation.scale.z = 0.1
			self.orientation.color.a = 1.0
			self.orientation.color.r = 0.0
			self.orientation.color.g = 0.0
			self.orientation.color.b = 1.0

			# q = [q.x, q.y, q.z, q.w]
			_, _, theta = euler_from_quaternion(q)
			self.xhat = np.array([x, y, v, theta, omega])
			self.xhat_integrated = np.array([x, y, v, theta, omega])
			# self.P = np.identity(5)
			self.P = np.zeros((5, 5))

			self.start_time = rospy.get_time()
			self.prev_time = self.start_time
			self.prev_time2 = self.start_time
			self.initialized2 = True
		elif self.initialized1 and self.initialized2:
			# self.prev_time = rospy.get_time()
			v = data.twist.twist.linear.x
			omega = data.twist.twist.angular.z
			self.y = np.array([v, omega])
			# self.time.append(rospy.get_time() - self.start_time)
			# self.v_history.append(v)
			# self.omega_history.append(omega)
			self.predict()
			self.integrate()

	def integrate(self):
		self.current_time2 = rospy.get_time()
		self.xhat_integrated = integrate_dynamics(self.xhat_integrated, self.current_time2-self.prev_time2, self.y)
		self.time_integrated.append(self.current_time2 - self.start_time)
		self.prev_time2 = self.current_time2
		self.x_integrated.append(self.xhat_integrated[0])
		self.y_integrated.append(self.xhat_integrated[1])
		# self.v_history.append(self.xhat[2])
		self.theta_integrated.append(self.xhat_integrated[3])
		# self.omega_history.append(self.xhat[4])
		# self.publish_state_estimate()

	def filtered_odom_callback(self, data):
		self.x = data.pose.pose.position.x
		self.y_filtered = data.pose.pose.position.y
		q = data.pose.pose.orientation
		self.theta = euler_from_quaternion([q.x, q.y, q.z, q.w])[2]
		self.v = data.twist.twist.linear.x
		self.omega = data.twist.twist.angular.z
		self.initialized1 = True
		if self.initialized1 and self.initialized2:
			self.current_time3 = rospy.get_time()
			# self.xhat = self.f(self.xhat, self.current_time-self.prev_time)
			self.time_actual.append(self.current_time3 - self.start_time)
			# self.prev_time = self.current_time
			self.x_actual.append(self.x)#self.xhat[0]-self.x)
			self.y_actual.append(self.y_filtered)#self.xhat[1]-self.y_filtered)
			# self.v_history.append(self.xhat[2])
			self.theta_actual.append(self.theta)#self.xhat[3]-self.theta)
			# self.omega_history.append(self.xhat[4])

		# 	# self.publish_state_estimate()

	def scan_callback(self, data):
		pass

	def mahalanobis_norm(self, data):
		pass

	def predict(self):
		self.current_time = rospy.get_time()
		F = self.F(self.xhat, self.current_time-self.prev_time)
		self.xhat = self.f(self.xhat, self.current_time-self.prev_time)
		self.prev_time = self.current_time
		self.P = np.dot(np.dot(F, self.P), F.transpose()) + self.V # + (self.xhat*np.identity(5) - self.V)
		if self.i == 5:
			print('F', np.linalg.norm(F))
			print('xhat', self.xhat[0])
			print('yhat', self.xhat[1])
			print('vhat', self.xhat[2])
			print('thetahat', self.xhat[3])
			print('omegahat', self.xhat[4])
			print('P', self.P[0, 0], self.P[1, 1])
		self.update()

	def update(self):
		H = self.H(self.xhat)
		nu = self.y - self.h(self.xhat) # not sure if it's important here to include the process and measurement noise, or if xhat is supposed to be something else
		S = np.dot(np.dot(H, self.P), H.transpose()) + self.W # + (self.h(self.xhat)*np.identity(2) - self.W)
		R = np.dot(np.dot(self.P, H.transpose()), np.linalg.inv(S))
		self.xhat = self.xhat + np.dot(R, nu)
		self.P = self.P - np.dot(np.dot(R, H),self.P)
		if self.i == 5:
			print('H', np.linalg.norm(H))
			print('nu', np.linalg.norm(nu))
			print('S', np.linalg.norm(S))
			print('R', np.linalg.norm(R))
			print('R nu', np.linalg.norm(np.dot(R, nu)))
			print('RHP', np.linalg.norm(np.dot(np.dot(R, H),self.P)))
		self.publish_state_estimate()

	def publish_state_estimate(self):
		if self.i == 5:
			print('xhat: ', self.xhat[0], 'x: ', self.x)
			print('yhat: ', self.xhat[1], 'y: ', self.y_filtered)
			print('vhat: ', self.xhat[2], 'v: ', self.v)
			print('thetahat: ', self.xhat[3], 'theta: ', self.theta)
			print('omegahat: ', self.xhat[4], 'omega: ', self.omega)
			# print('v odom: ', self.y[0])
			# print('omega odom: ', self.y[1])
			# print('innovation: ', self.xhat[2] - self.y[0], self.xhat[4] - self.y[1])
			print('P: ', self.P[0, 0], self.P[1, 1])
			print()
			self.i = 0
		else:
			self.i += 1

		self.time.append(self.current_time - self.start_time)
		self.x_history.append(self.xhat[0])#-self.x)
		self.y_history.append(self.xhat[1])#-self.y_filtered)
		# self.v_history.append(self.xhat[2])
		self.theta_history.append(self.xhat[3])#-self.theta)
		self.omega_history.append(self.xhat[4])

		self.position.header.stamp = rospy.Time.now()
		self.position.pose.position.x = self.xhat[0]
		self.position.pose.position.y = self.xhat[1]
		self.position.scale.x = self.P[0, 0]
		self.position.scale.y = self.P[1, 1]

		self.orientation.header.stamp = rospy.Time.now()
		self.orientation.pose.position.x = self.xhat[0]
		self.orientation.pose.position.y = self.xhat[1]
		theta = self.xhat[3]
		q = quaternion_from_euler(0, 0, theta)
		self.orientation.pose.orientation.x = q[0]
		self.orientation.pose.orientation.y = q[1]
		self.orientation.pose.orientation.z = q[2]
		self.orientation.pose.orientation.w = q[3]

		self.position_pub.publish(self.position)
		self.orientation_pub.publish(self.orientation)
		if self.time[-1] > 30:
			if self.plot:
				plt.figure()
				plt.plot(self.time, self.x_history, 'r')
				plt.plot(self.time_integrated, self.x_integrated, 'g')
				plt.plot(self.time_actual, self.x_actual, 'b')
				plt.legend(['kalman filter', 'integration', 'actual'])
				plt.xlabel('Time(s)')
				plt.ylabel('x position(m)')
				plt.title('x vs Time')

				plt.figure()
				plt.plot(self.time, self.y_history, 'r')
				plt.plot(self.time_integrated, self.y_integrated, 'g')
				plt.plot(self.time_actual, self.y_actual, 'b')
				plt.legend(['kalman filter', 'integration', 'actual'])
				plt.xlabel('Time(s)')
				plt.ylabel('y position(m)')
				plt.title('y vs Time')

				plt.figure()
				plt.plot(self.time, self.theta_history, 'r')
				plt.plot(self.time_integrated, self.theta_integrated, 'g')
				plt.plot(self.time_actual, self.theta_actual, 'b')
				plt.legend(['kalman filter', 'integration', 'actual'])
				plt.xlabel('Time(s)')
				plt.ylabel('orientation theta(rad)')
				plt.title('Orientation vs Time')

				# plt.figure()
				# plt.plot(self.time, sorted(self.x_history) ,'r')
				# mean = np.mean(self.x_history)
				# var = np.var(self.x_history)
				# print(var)
				# plt.plot(self.time, sorted(np.random.normal(mean, np.sqrt(var), size=(len(self.time),)).tolist()), 'b')
				# plt.legend(['data', 'normal'])
				# plt.title('x Error vs Time')
				# plt.xlabel('Time(s)')
				# plt.ylabel('Error(m)')

				# plt.figure()
				# plt.plot(self.time, sorted(self.y_history), 'r')
				# mean = np.mean(self.y_history)
				# var = np.var(self.y_history)
				# print(var)
				# plt.plot(self.time, sorted(np.random.normal(mean, np.sqrt(var), size=(len(self.time),)).tolist()), 'b')
				# plt.legend(['data', 'normal'])
				# plt.title('y Error vs Time')
				# plt.xlabel('Time(s)')
				# plt.ylabel('Error(m)')

				# # plt.figure()
				# # # plt.plot(self.time, sorted(self.v_history), 'r')
				# # plt.plot(self.time, self.v_history)
				# # # mean = np.mean(self.v_history)
				# # # var = np.var(self.v_history)
				# # # print(var)
				# # # plt.plot(self.time, sorted(np.random.normal(mean, np.sqrt(var), size=(len(self.time),)).tolist()), 'b')
				# # # plt.legend(['data', 'normal'])
				# # plt.title('Velocity from Odometry vs Time')
				# # plt.xlabel('Time(s)')
				# # plt.ylabel('Velocity(m/s)')

				# plt.figure()
				# plt.plot(self.time, sorted(self.theta_history), 'r')
				# mean = np.mean(self.theta_history)
				# var = np.var(self.theta_history)
				# print(var)
				# plt.plot(self.time, sorted(np.random.normal(mean, np.sqrt(var), size=(len(self.time),)).tolist()), 'b')
				# plt.legend(['data', 'normal'])
				# plt.title('theta Error vs Time')
				# plt.xlabel('Time(s)')
				# plt.ylabel('Error(rad)')

				# # plt.figure()
				# # # plt.plot(self.time, sorted(self.omega_history), 'r')
				# # plt.plot(self.time, self.omega_history)
				# # # mean = np.mean(self.omega_history)
				# # # var = np.var(self.omega_history)
				# # # print(var)
				# # # plt.plot(self.time, sorted(np.random.normal(mean, np.sqrt(var), size=(len(self.time),)).tolist()), 'b')
				# # # plt.legend(['data', 'normal'])
				# # plt.title('Angular Velocity from Odometry vs Time')
				# # plt.xlabel('Time(s)')
				# # plt.ylabel('Angular Velocity(rad/s)')

				plt.show()
				self.plot = False

# 5 seconds
# 0.0182560775451
# 0.0808943525884
# 0.0218975421455
# 0.0891248631988
# 0.0218975421455

# 10 seconds
# 0.134196269236
# 0.178253507059
# 0.02709973916
# 0.729888709773
# 0.02709973916

class KalmanFilterSLAM:

	def __init__(self):
		pass

	def odom_callback(self, data):
		pass

	def scan_callback(self, data):
		pass

	def mahalanobis_norm(self, data):
		pass

	def predict(self):
		pass

	def update(self):
		pass

	def publish_state_estimate(self):
		pass

	def publish_map(self):
		pass

if __name__ == '__main__':
	rospy.init_node("kalman_filter")
	EKF = KalmanFilter(dynamics, measurement_model, linear_dynamics, linear_measurement_model)
	rospy.spin()