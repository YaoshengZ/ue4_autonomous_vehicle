#include <iostream>
#include <fstream>
#include <string>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "opencv2/imgcodecs.hpp"

#include <communication/multi_socket.h>
#include <models/tronis/ImageFrame.h>
#include <grabber/opencv_tools.hpp>
#include <models/tronis/BoxData.h>

#define _USE_MATH_DEFINES
#include <math.h>

using namespace std;
using namespace cv;

class LaneAssistant
{
	public:
		LaneAssistant()
		{
		}

		bool processData( tronis::CircularMultiQueuedSocket& socket )
        {
            socket.send( tronis::SocketData( "Speed: " + to_string( ego_velocity_ ) ) );  // send ego speed via socket
            getSteeringInput( socket );                                                   // send steering value via socket
            getThrottleInput( socket );                                                   // send throttle value via socket
			return true;
		}

	protected:
		std::string image_name_;
		cv::Mat image_;
        tronis::LocationSub ego_location_;
        tronis::OrientationSub ego_orientation_;
        tronis::BoxDataSub boxes_;
        double ego_velocity_;

		//++++++++++++++++++++++red light detection parameters+++++++++++++++++++++++
        cv::Mat redlightraw;
		cv::Mat red_light_img;
        cv::Mat redlight_roi_img;

	    //+++++++++++++++++++++++++lane detection parameters++++++++++++++++++++++++
		Point ego_leftS, ego_leftE;     // ego left lane start and end point
        Point ego_rightS, ego_rightE;   // ego right lane start and end point
        Point directionS, directionE;   // car driving direction
        double rows = 512, cols = 720;  // original Picture size (720, 512).

		//++++++++++++++++++++++++steering control parameters+++++++++++++++++++++++
        double steering_norm = 0;       // normalized steering input value
        float STEER_P = 1;              // P-Factor for STEER PID controller
        float STEER_I = 0.00025;        // I-Factor for STEER PID controller
        float STEER_D = 0;              // D-Factor for STEER PID controller
        float STEER_MAX = 256;          // max pixel diff between x_curr and x_target (512/2 for max) to normalize
        double steer_error_P_old = 0;   // steering P at t-1
        double steer_error_I_sum = 0;   // steering I sum

		//++++++++++++++++++++++throttle control parameters++++++++++++++++++++++++++
        double throttle_norm = 0;       // normalized throttle input value
        double VEL_TAR = 60;            // Max velocity without preceeding vehicle
        int ACC_P = 45;                 // P-Factor for ACCELERATION PID controller
        int ACC_I = 125;                // I-Factor for ACCELERATION PID controller
        int ACC_D = 0;                  // D-Factor for ACCELERATION PID controller
        float DIST_ACT = 100;           // Distance at which an object becomes relevant for ACC in meters
        double DIST_TAR_SLOW = 7;       // target distance to preceeding vehicle
        int DIST_P = 15;                // P-Factor for DIST PID controller
        int DIST_I = 15;                // I-Factor for DIST PID controller
        int DIST_D = 0;                 // D-Factor for DIST PID controller
        int DIST_WATCHDOG_MAX = 150;    // checks if tronis still sends distance updates
        double vel_error_P_old = 0;     // velocity P at t-1
        double vel_error_I_sum = 0;     // velocity I sum
        double dist_curr = 101;         // current distance to preceeding vehicle
        double dist_curr_old = 0;       // distance at t-1
        double dist_error_P_old = 0;    // to check if tronis still updates boxes
        double dist_error_I_sum = 0;    // distance I sum
        int dist_watchdog = 0;          // to check if tronis still updates boxes
        double vel_tar = 0;             // calculated target velocity
        double t_dist_diff = 0;         // time since last function call
        chrono::time_point<chrono::steady_clock> t_dist_start;  // stop watch
        chrono::time_point<chrono::steady_clock> t_dist_stop;

		//--------------------red light detection----------------------
		void redLightDetection()
        {
            // 1. set a red hls detector
            cv::Mat kernel = getStructuringElement( MORPH_RECT, Size( 3, 3 ) );
            cvtColor( image_, redlightraw, cv::COLOR_BGR2HLS );
            Scalar red_lower( 0, 0, 50 );
            Scalar red_upper( 8, 255, 255 );
            inRange( redlightraw, red_lower, red_upper, red_light_img );
            red_light_img.convertTo( red_light_img, CV_8UC1 );
            dilate( red_light_img, red_light_img, kernel );

			// 2. make a red light roi mask 
            cv::Mat redlightmask = Mat::zeros( image_.size(), red_light_img.type() );
            const int num = 4;
            Point points[1][num] = {Point( cols * 0.4, 0 ),          
				                    Point( cols * 0.4, rows * 0.35 ),
                                    Point( cols * 0.6, rows * 0.35 ), 
				                    Point( cols * 0.6, 0 )};
            const Point* polygon = points[0];
            fillConvexPoly( redlightmask, polygon, num, Scalar( 255 ) );

			// 3. use red light roi mask to filter red light img
            cv::bitwise_and( red_light_img, redlightmask, redlight_roi_img );
            //imshow( "Red Light Detection roi", redlight_roi_img );
        }
		
		//----------------------lane detection------------------------
        vector<Vec4d> setLanes()
        {
            redLightDetection();

			// 1a-1. from tronis image to gaussian blurred image
			cv::Mat blur_img; 
            GaussianBlur( image_, blur_img, Size( 3, 3 ), 0, 0, BORDER_DEFAULT );

			// 1a-2. from gaussian blurred image to grayscale image
            cv::Mat gray_img;
            cvtColor( blur_img, gray_img, cv::COLOR_BGR2GRAY );
            
			// from grayscale image to binary image
            //cv::Mat binary_img; 
            //cv::threshold( gray_img, binary_img, 190, 255, cv::THRESH_BINARY );
			
			// 1a-3. from binary image to edge image
            cv::Mat edge_img; 
            Canny( gray_img, edge_img, 200, 250 );
            cv::Mat kernel = getStructuringElement( MORPH_RECT, Size( 3, 3 ) );    
            dilate( edge_img, edge_img, kernel );    // dilation of edges
            
			// 1b-1. from tronis image to white hls image
            cv::Mat hlsraw;
            cv::Mat white_hls_img; 
			cvtColor( image_, hlsraw, cv::COLOR_BGR2HLS );
            Scalar white_lower( 0, 100, 0 );
            Scalar white_upper( 180, 155, 255 );
            inRange( hlsraw, white_lower, white_upper, white_hls_img );
            white_hls_img.convertTo( white_hls_img, CV_8UC1 );
            dilate( white_hls_img, white_hls_img, kernel );    // dilation of white hls image

			// 1b-2. from tronis image to yellow hls image
            cv::Mat yellow_hls_img; 
			cvtColor( image_, hlsraw, cv::COLOR_BGR2HLS );
            Scalar yellow_lower( 10, 0 ,75 );
            Scalar yellow_upper( 23, 255, 255 );
            inRange( hlsraw, yellow_lower, yellow_upper, yellow_hls_img );
            yellow_hls_img.convertTo( yellow_hls_img, CV_8UC1 );
            dilate( yellow_hls_img, yellow_hls_img, kernel );    // dilation of yellow hls image

			// 1b-3. choose between white hls image and yellow hls image
            cv::Mat hls_img;
            cv::bitwise_or( yellow_hls_img, white_hls_img, hls_img );
			/*
            hls_img = white_hls_img;
			if (countNonZero(white_hls_img) < countNonZero(yellow_hls_img))
			{
                hls_img = yellow_hls_img;
			}
			*/

			// 2. combine edge image with hls image to get edge hls image
            cv::Mat edge_hls_img; 
            cv::bitwise_and( edge_img, hls_img, edge_hls_img );

            // 3. make a roi mask
            cv::Mat mask = Mat::zeros( image_.size(), edge_hls_img.type() );
            const int num = 8;
            Point points[1][num] = {Point( 0, rows * 0.95 ),
                                    Point( 0, rows * 0.65 ),
                                    Point( cols * 0.4, rows * 0.55 ), 
				                    Point( cols * 0.6, rows * 0.55 ),
                                    Point( cols, rows * 0.65 ),
                                    Point( cols, rows * 0.95 ),
                                    Point( cols * 0.7, rows * 0.78 ),
			                        Point( cols * 0.3, rows * 0.78 )};
            const Point* polygon = points[0];
            fillConvexPoly( mask, polygon, num, Scalar( 255 ) );

			// 4. from edge hsl image to roi image
            cv::Mat roi_img;
            cv::bitwise_and( edge_hls_img, mask, roi_img );
            imshow( "Region of Interest", roi_img );    // show the roi image

			// 5. from roi image to get hough lines set
            vector<Vec4d> raw_lanes;                                         // will hold all the results of the detection
            HoughLinesP( roi_img, raw_lanes, 1, CV_PI / 180, 100, 25, 25 );   // Probabilistic Line Transform

            return raw_lanes;
        }

        void getLanes( vector<Vec4d> raw_lanes )
        {
            vector<Vec4d> left_lanes, right_lanes;
            Vec4f left_lane_function, right_lane_function;
            vector<Point> left_points, right_points;

            ego_leftS.y = 300;
            ego_rightS.y = 300;
            ego_leftE.y = 500;
            ego_rightE.y = 500;

            double left_k, right_k;  // gradient
            Point left_b, right_b;

            for( auto lane : raw_lanes )  // divide the line set into left and right part based on the line center point
            {
                double lane_center = ( lane[0] + lane[2] ) / 2;

                if( lane_center < cols / 2 )
                {
                    left_lanes.push_back( lane );
                }
                else
                {
                    right_lanes.push_back( lane );
                }
            }

            // get the left lines
            for( auto left_lane : left_lanes )  // add all the points into a vector
            {
                left_points.push_back( Point( left_lane[0], left_lane[1] ) );
                left_points.push_back( Point( left_lane[2], left_lane[3] ) );
            }
            if( left_points.size() > 0 )  // fit a line with the method of least square
            {
                // fitLine(input vector, output line, distance type, distance parameter, radial
                // parameter, angle parameter) output (vx, vy, x, y)
                cv::fitLine( left_points, left_lane_function, cv::DIST_L2, 0, 0.01, 0.01 );

                left_k = left_lane_function[1] / left_lane_function[0];
                left_b = Point( left_lane_function[2], left_lane_function[3] );

                ego_leftS.x = ( ego_leftS.y - left_b.y ) / left_k + left_b.x;
                ego_leftE.x = ( ego_leftE.y - left_b.y ) / left_k + left_b.x;
            }

            // get the right lines
            for( auto right_lane : right_lanes )
            {
                right_points.push_back( Point( right_lane[0], right_lane[1] ) );
                right_points.push_back( Point( right_lane[2], right_lane[3] ) );
            }
            if( right_points.size() > 0 )
            {
                cv::fitLine( right_points, right_lane_function, cv::DIST_L2, 0, 0.01, 0.01 );

                right_k = right_lane_function[1] / right_lane_function[0];
                right_b = Point( right_lane_function[2], right_lane_function[3] );

                ego_rightS.x = ( ego_rightS.y - right_b.y ) / right_k + right_b.x;
                ego_rightE.x = ( ego_rightE.y - right_b.y ) / right_k + right_b.x;
            }

            directionS = ( ego_leftS + ego_rightS ) / 2;
            directionE = ( ego_leftE + ego_rightE ) / 2;
            // cv::Vec4d direction( directionS.x, directionS.y, directionE.x, directionE.y );
        }

        void detectLanes()  // Function to detect lanes based on camera image
        {
            vector<Vec4d> raw_lanes = setLanes();
            // vector<Vec4d> warning_lanes = setWarnings();
            getLanes( raw_lanes );

            // Draw the lane lines and show results
            line( image_, ego_leftS, ego_leftE, Scalar( 255, 0, 0 ), 3, LINE_AA );
            line( image_, ego_rightS, ego_rightE, Scalar( 255, 0, 0 ), 3, LINE_AA );

            // Draw the driving direction lines and show results
            line( image_, Point( directionS.x, directionS.y ), Point( directionE.x, directionE.y ),
                  Scalar( 0, 255, 0 ), 3, LINE_AA );
        }

		//---------------------- Steering control--------------------
		void setSteeringInput()
        {
			double steer_error_P = directionS.x - cols / 2; // use directionS.x --> react earlier + smoother, use directionE.x --> react later + jerky 
            double steer_error_D = steer_error_P - steer_error_P_old;
            double steer_tar = STEER_P * steer_error_P + STEER_I * steer_error_I_sum + STEER_D * steer_error_D;
            steer_error_P_old = steer_error_P;
            steer_error_I_sum += steer_error_P;
			steering_norm = 2 * ( steer_tar + STEER_MAX ) / ( 2 * STEER_MAX ) - 1; 
            if( steering_norm > 1 )                         // normalize the steering input to between -1 and 1
            {
                steering_norm = 1;
            }
            else if( steering_norm < -1 )
            {
                steering_norm = -1;
            }
        }

		void getSteeringInput( tronis::CircularMultiQueuedSocket& socket )
        {
            setSteeringInput();
            string prefix_steering = "Steering: ";
            socket.send( tronis::SocketData( prefix_steering + to_string( steering_norm ) ) );
        }

		//---------------------throttle control-------------------------
        void accelerationControl()
        {
            double vel_curr = ego_velocity_ * ( 36. / 1000. );  // from cm/s to km/h
            if( dist_curr < DIST_ACT )  // acc function
            {
                // stop stopwatch and take time for integral and derivative
                t_dist_stop = chrono::steady_clock::now();
                t_dist_diff =
                    chrono::duration_cast<chrono::milliseconds>( t_dist_stop - t_dist_start )
                        .count();

                double dist_tar = 0.5 * vel_curr;  // set target distance to half of current speed, at least 7m
                if( vel_curr < 14 )
                {
                    dist_tar = DIST_TAR_SLOW;
                }

                double dist_error_P = dist_curr - dist_tar;
                double dist_error_D = ( dist_error_P - dist_error_P_old ) / t_dist_diff;

                if( t_dist_diff < 100 )   // avoid irrational time measurements
                {
                    double dist_error_I_next = ( (double)DIST_I / 1e6 ) *
                                               ( dist_error_I_sum + dist_error_P * t_dist_diff );
                    // allow the I part to only influence +- 2kph and include time since last call
                    if( dist_error_I_next <= 2 )
                    {
                        dist_error_I_sum += dist_error_P * t_dist_diff;
                    }
                }

                // PID controller for target velocity with vel_curr aus offset
                vel_tar = vel_curr + dist_error_P * ( (double)DIST_P / 10 ) +
                          dist_error_I_sum * ( (double)DIST_I / 1e6 ) +
                          dist_error_D * ( (double)DIST_D / 1e6 );

                ///// in braking scenarios, remove the offset again, including hysteresis
                //         if( ( dist_curr < DIST_TAR_SLOW || abs( dist_curr - DIST_TAR_SLOW ) < 3 )
                //         )
                //         {
                //             vel_tar -= vel_curr;
                //         }

                if( vel_tar < 1 )    // avoid negative target velocities
                {
                    vel_tar = 0;
                }

                // reduce I when driving slow and getting close to the target vehicle
                if( ( dist_curr < DIST_TAR_SLOW || abs( dist_curr - DIST_TAR_SLOW ) < 3 ) &&
                    vel_tar < 5 )
                {
                    dist_error_I_sum *= 0.9;
                    cout << "dist I reduction" << endl;
                }

                // no new distance updates from tronis = reset to CC
                dist_curr_old = dist_curr;
                if( dist_curr == dist_curr_old )
                {
                    dist_watchdog++;
                    // cout << " WD+1: " << dist_watchdog << endl;
                    if( dist_watchdog >= DIST_WATCHDOG_MAX )
                    {
                        dist_curr = DIST_ACT + 1;
                        dist_watchdog = 0;
                    }
                }
                // start stopwatch
                t_dist_start = chrono::steady_clock::now();

                cout << "distance: " << dist_curr << " m, target: " << dist_tar
                     << " m || P = " << ( (double)DIST_P / 10 )
                     << " == " << dist_error_P * ( (double)DIST_P / 10 )
                     << ", I = " << ( (double)DIST_I / 1e6 )
                     << " == " << dist_error_I_sum * ( (double)DIST_I / 1e6 )
                     << ", D = " << ( (double)DIST_D / 1e6 )
                     << " == " << dist_error_D * ( (double)DIST_D / 1e6 ) << " || cmd: " << vel_tar
                     << " kmh" << endl;

                // call cruiseControl to apply calculated target velocity
                velocityControl( vel_curr, vel_tar, true );
            }
            else
            {
                // velocity controller
                velocityControl( vel_curr, vel_tar, false );
            }
        }
        void velocityControl( double vel_c, double vel_t, bool acc_flag )
        {
			// if acc is active, then limit the target velocity by VEL_TAR (e.g. speed signs)
            if( acc_flag )
            {				
				if( VEL_TAR < vel_t )
                {
                    vel_t = VEL_TAR;
                }
            }
            else
            {
                vel_t = VEL_TAR;
            }
            // similar to acceleration PID control
            double vel_error_P = vel_t - vel_c;
            double vel_error_D = ( vel_error_P - vel_error_P_old );
            double acc_tar = ( (double)ACC_P / 1000 ) * vel_error_P +
                             ( (double)ACC_I / 1000000 ) * vel_error_I_sum +
                             ( (double)ACC_D / 1000 ) * vel_error_D;
            vel_error_P_old = vel_error_P;
            // no multiplication with time, bc controller is stable as is
            vel_error_I_sum += vel_error_P;

            if( dist_curr < 5 )    // below 5m, brake hard
            {
                acc_tar = -1;
                cout << "hard brake" << endl;
            }

            // reduce I when driving slow and getting close to the target vehicle
            if( ( dist_curr < DIST_TAR_SLOW || abs( dist_curr - DIST_TAR_SLOW ) < 3 ) &&
                vel_tar < 5 )
            {
                vel_error_I_sum *= 0.;
                cout << "vel I  reduction" << endl;

                if( vel_c < 1 )    // avoid slow rolling
                {
                    acc_tar = 0;
                }
            }

            if( vel_c < 2 && acc_tar < 0 )  // avoid driving backwards
            {
                acc_tar = 0;
                cout << "avoid backwards driving" << endl;
            }

            throttle_norm = acc_tar;
            /*
			if ( countNonZero( redlight_roi_img) > 5 )  // stop at the red light
			{
				if (vel_c > 10)
				{
                     throttle_norm = -0.5;
				}
				else if (0 < vel_c <= 10)
				{
                     throttle_norm = 0;
				}
                cout << "stop at red light" << endl;
			}
			*/
			if ( vel_c > 45 && abs(steering_norm) > 0.1 )  // slow down at sharp curves
			{
                throttle_norm = 0;
                cout << "slow down at sharp curves" << endl;
			}
			
            if( throttle_norm > 1 )    // normalize the throttle input to between -1 and 1
            {
                throttle_norm = 1;
            }
            else if( throttle_norm < -1 )
            {
                throttle_norm = -1;
            }
            cout << "velocity: " << vel_c << " kmh, target: " << vel_t
                 << " kmh || P = " << ( (double)ACC_P / 1000 )
                 << " == " << vel_error_P * ( (double)ACC_P / 1000 )
                 << ", I = " << ( (double)ACC_I / 1000000 )
                 << " == " << vel_error_I_sum * ( (double)ACC_I / 1000000 )
                 << ", D = " << ( (double)ACC_D / 1000 )
                 << " == " << vel_error_D * ( (double)ACC_D / 1000 ) << " || cmd = " << throttle_norm
                 << endl;
        }

		void getThrottleInput( tronis::CircularMultiQueuedSocket& socket )
        {
            string prefix_throttle = "Throttle: ";
            socket.send( tronis::SocketData( prefix_throttle + to_string( throttle_norm ) ) );
        }

// Helper functions, no changes needed
    public:
		// Function to process received tronis data
		bool getData( tronis::ModelDataWrapper data_model )
		{
            if( data_model->GetModelType() == tronis::ModelType::Tronis )
            {
                //std::cout   << "Id: " << data_model->GetTypeId()
                //            << ", Name: " << data_model->GetName()
                //            << ", Time: " << data_model->GetTime() << std::endl;

                // if data is sensor output, process data
                switch( static_cast<tronis::TronisDataType>( data_model->GetDataTypeId() ) )
                {
                    case tronis::TronisDataType::Image:
                    {
                        processImage(
                            data_model->GetName(),
                            data_model.get_typed<tronis::ImageSub>()->Image );
                        break;
                    }
                    case tronis::TronisDataType::ImageFrame:
                    {
                        const tronis::ImageFrame& frames(
                            data_model.get_typed<tronis::ImageFrameSub>()->Images );
                        for( size_t i = 0; i != frames.numImages(); ++i )
                        {
                            std::ostringstream os;
                            os << data_model->GetName() << "_" << i + 1;

                            processImage( os.str(), frames.image( i ) );
                        }
                        break;
                    }
                    case tronis::TronisDataType::ImageFramePose:
                    {
                        const tronis::ImageFrame& frames(
                            data_model.get_typed<tronis::ImageFramePoseSub>()->Images );
                        for( size_t i = 0; i != frames.numImages(); ++i )
                        {
                            std::ostringstream os;
                            os << data_model->GetName() << "_" << i + 1;

                            processImage( os.str(), frames.image( i ) );
                        }
                        break;
                    }
                    case tronis::TronisDataType::PoseVelocity:
                    {
                        processPoseVelocity( data_model.get_typed<tronis::PoseVelocitySub>() );
                        accelerationControl();
                        break;
                    }
                    case tronis::TronisDataType::BoxData:
                    {
                        processBox( data_model.get_typed<tronis::BoxDataSub>() );
						//processObject( data_model.get_typed<tronis::BoxDataSub>() );
                        // std::cout << data_model.get_typed<tronis::BoxDataSub>()->ToString() <<std::endl;
                        break;
                    }
                    default:
                    {
                        //std::cout << data_model->ToString() << std::endl;
                        break;
                    }
                }
                return true;
            }
            else
            {
                //std::cout << data_model->ToString() << std::endl;
                return false;
            }
		}

	protected:
		// Function to show an openCV image in a separate window
        void showImage( std::string image_name, cv::Mat image )
        {
            cv::Mat out = image;
            if( image.type() == CV_32F || image.type() == CV_64F )
            {
                cv::normalize( image, out, 0.0, 1.0, cv::NORM_MINMAX, image.type() );
            }
            cv::namedWindow( image_name.c_str(), cv::WINDOW_NORMAL );
            cv::imshow( image_name.c_str(), out );
        }

		// Function to convert tronis image to openCV image
		bool processImage( const std::string& base_name, const tronis::Image& image )
        {
            std::cout << "processImage" << std::endl;
            if( image.empty() )
            {
                std::cout << "empty image" << std::endl;
                return false;
            }

            image_name_ = base_name;
            image_ = tronis::image2Mat( image );

            detectLanes();
            showImage( image_name_, image_ );

            return true;
        }
        // Function to convert tronis velocity to processible format
        bool processPoseVelocity( tronis::PoseVelocitySub* msg )
        {
            ego_location_ = msg->Location;
            ego_orientation_ = msg->Orientation;
            ego_velocity_ = msg->Velocity;
            return true;
        }
        // Function to convert tronis bounding boxes to processible format
        bool processBox( tronis::BoxDataSub* msg )
        {
            vector<string> box_names;
            vector<double> box_distances;
            // loop through all detected boxes
            for( int i = 0; i < msg->Objects.size(); i++ )
            {
                // std::cout << msg->ToString() << std::endl;
                tronis::ObjectSub& box = msg->Objects[i];

                // filter for right object size
                // if( box.BB.Extends.X > 100 && box.BB.Extends.X < 400 && box.BB.Extends.Y > 100 &&
                //    box.BB.Extends.Y < 300 )
                if( box.BB.Extends.X > 100 && box.BB.Extends.X < 800 && box.BB.Extends.Y > 100 &&
                    box.BB.Extends.Y < 800 )
                {
                    // remove own vehicle from possibilities
                    if( box.Pose.Location.X != 0.0 )
                    {
                        // remove vehicles from parallel lanes
                        if( abs( box.Pose.Location.Y ) < 400 )
                        {
                            // cout << box.ActorName.Value() << ", is " << _hypot(
                            // box.Pose.Location.X / 100, box.Pose.Location.Y / 100 ) << " m ahead."
                            // << endl;
                            double dist_curr_temp =
                                _hypot( box.Pose.Location.X / 100, box.Pose.Location.Y / 100 );
                            if( dist_curr_temp > 5 )
                            {
                                // compensate center position of box sensor: 2.5 (own) + 2.5m
                                // (preceeding car)
                                dist_curr_temp -= 5;
                            }
                            // append to vectors
                            box_names.push_back( box.ActorName.Value() );
                            box_distances.push_back( dist_curr_temp );
                        }
                    }
                }
            }
            // find minimum distance box
            double box_min_it = -1;
            double box_min = 100;
            for( int i = 0; i < box_names.size(); i++ )
            {
                // cout << "Box " << i << ": " << box_names[i] << " (" << box_distances[i] << "m)"
                // << endl;
                if( box_distances[i] < box_min )
                {
                    box_min = box_distances[i];
                    box_min_it = i;
                }
            }
            // use min distance box for distance control (in case there are multiple cars)
            if( box_min_it != -1 )
            {
                // cout << "Target Box " << box_min_it << ": " << box_names[box_min_it] << " (" <<
                // box_distances[box_min_it] << "m)" << endl;
                dist_curr = box_distances[box_min_it];
            }

            return true;
        }
};

// main loop opens socket and listens for incoming data
int main( int argc, char** argv )
{
    std::cout << "Welcome to lane assistant" << std::endl;

	// specify socket parameters
	std::string socket_type = "TcpSocket";
    std::string socket_ip = "127.0.0.1";
    std::string socket_port = "7778";

    std::ostringstream socket_params;
    socket_params << "{Socket:\"" << socket_type << "\", IpBind:\"" << socket_ip << "\", PortBind:" << socket_port << "}";

    int key_press = 0;	// close app on key press 'q'
    tronis::CircularMultiQueuedSocket msg_grabber;
    uint32_t timeout_ms = 500; // close grabber, if last received msg is older than this param

	LaneAssistant lane_assistant;

	while( key_press != 'q' )
    {
        std::cout << "Wait for connection..." << std::endl;
        msg_grabber.open_str( socket_params.str() );

        if( !msg_grabber.isOpen() )
        {
            printf( "Failed to open grabber, retry...!\n" );
            continue;
        }

        std::cout << "Start grabbing" << std::endl;
		tronis::SocketData received_data;
        uint32_t time_ms = 0;

        while( key_press != 'q' )
        {
			// wait for data, close after timeout_ms without new data
            if( msg_grabber.tryPop( received_data, true ) )
            {
				// data received! reset timer
                time_ms = 0;

				// convert socket data to tronis model data
                tronis::SocketDataStream data_stream( received_data );
                tronis::ModelDataWrapper data_model(
                    tronis::Models::Create( data_stream, tronis::MessageFormat::raw ) );
                if( !data_model.is_valid() )
                {
                    std::cout << "received invalid data, continue..." << std::endl;
                    continue;
                }
				// identify data type
                lane_assistant.getData( data_model );
                lane_assistant.processData( msg_grabber );
            }
            else
            {
				// no data received, update timer
                ++time_ms;
                if( time_ms > timeout_ms )
                {
                    std::cout << "Timeout, no data" << std::endl;
                    msg_grabber.close();
                    break;
                }
                else
                {
                    std::this_thread::sleep_for( std::chrono::milliseconds( 10 ) );
                    key_press = cv::waitKey( 1 );
                }
            }
        }
        msg_grabber.close();
    }
    return 0;
}
