function [locations] = generate_mosquito_data(image, num_seconds, frames_per_movement, numMos, radii, moveSizes)
                            
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

% example usage: 
%   image = imread('room.jpg');
%   generate_mosquito_data(image, 10, 3, 3, [3, 5, 6], [5, 5, 6]);

framerate = 20;
writerObj = VideoWriter('myVideo.avi');
writerObj.FrameRate = framerate;

 % open the video writer
 open(writerObj);
 
 trajs = zeros([numMos, 2]);
 positions = zeros([numMos, 2]);
 
 % initialize the mosquito positions
 for i=1:numMos
    positions(i, :) = round(rand([1,2]) .* [size(image, 1), size(image, 2)]);
    trajs(i, :) = rand([1,2]);
 end
 grids = [];
 % write the frames to the video
 for i = 1: frames_per_movement: (num_seconds * framerate)
     [positions, trajs] = generateMovement(positions, trajs, moveSizes, size(image));
     [grid, blendedImage] = insertMosquito(image, positions, radii);
     % convert the image to a frame
     frame = im2frame(blendedImage);
     
     for v=1:frames_per_movement
         grids = cat(4, grids, grid);
         writeVideo(writerObj, frame);
     end
 end
 % close the writer object
 close(writerObj);
 locations = grids;
 save('locations.mat', 'locations', '-v7.3');
end