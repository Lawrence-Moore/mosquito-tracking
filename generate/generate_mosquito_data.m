function generate_mosquito_data(image, num_seconds, frames_per_movement, pixelSize, moveSize)
                            
%                             video_path, new_video_name, num_seconds, mosquito_size, ...
%                                 mosquito_distance, mosquito_speed, frames_per_movement

% Takes in an image and generates synthetic training data for mosquito detection
%
% video_path - a string pointing to the video
% mosquito_size - size of the mosquito in cm
% mosquito_distance - distance from the camera to the mosquito (in m)
% mosquito_speed - speed the mosquito is moving in cm/s
% frames_per_movement - dictates at what rate the mosquito will move in
% video time

framerate = 5;
writerObj = VideoWriter('myVideo.avi');
writerObj.FrameRate = framerate;

 % open the video writer
 open(writerObj);
 
 traj = rand([1,2]);
 oldPosition = round(rand([1,2]) .* [size(image, 1), size(image, 2)]);
 grids = [];
 % write the frames to the video
 for i = 1: frames_per_movement: (num_seconds * framerate)
     [newPosition, traj] = generateMovement(oldPosition, traj, moveSize, size(image));
     [grid, blendedImage] = insertMosquito(image, newPosition, pixelSize);
     grids = cat(4, grids, grid);
     % convert the image to a frame
     frame = im2frame(blendedImage);
     
     for v=1:frames_per_movement 
         writeVideo(writerObj, frame);
     end
 end
 % close the writer object
 close(writerObj);

end