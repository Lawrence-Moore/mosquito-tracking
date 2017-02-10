function [locations] = generate_mosquito_data(img, ...
                                              num_seconds, frames_per_movement, ...
                                              numMos, radii, moveSizes,...
                                              video_name, mat_name, pixel_pos_name)
                            
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
%   generate_mosquito_data(img, 5, 3, 3, [3, 6, 7], [5, 5, 6], 'video.avi', 'loc1.mat');

framerate = 20;
writerObj = VideoWriter(video_name);
writerObj.FrameRate = framerate;

 % open the video writer
 open(writerObj);
 
 trajs = zeros([numMos, 2]);
 positions = zeros([numMos, 2]);
 
 all_positions = zeros([numMos, 2, 1]);
 all_positions(:, :, 1) = positions;
 
 % initialize the mosquito positions
 for i=1:numMos
    positions(i, :) = round(rand([1,2]) .* [size(img, 1), size(img, 2)]);
    trajs(i, :) = rand([1,2]);
 end
 grids = [];
 % write the frames to the video
 for i = 1: frames_per_movement: (num_seconds * framerate)
     [positions, trajs] = generateMovement(positions, trajs, moveSizes, size(img));
     [grid, blendedImage] = insertMosquito(img, positions, radii);
     % convert the image to a frame
     frame = im2frame(blendedImage);

     for v=1:frames_per_movement
         all_positions(:, :, v + i - 1) = positions; 
         grids = cat(4, grids, grid);
         writeVideo(writerObj, frame);
     end
 end
 % close the writer object
 close(writerObj);
 locations = grids;
 save(pixel_pos_name, 'all_positions', '-V7.3');
 save(mat_name, 'locations', '-V7.3');
end