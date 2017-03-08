num_images = 80;
start = 1;
for i = start: start + num_images - 1
   img_name = strcat(strcat('../data/image', num2str(i)), '.jpg');
   video_name =  strcat(strcat('../data/video', num2str(i)), '.avi');
   label_name =  strcat(strcat('../data/locations', num2str(i)), '.mat');
   pixel_pos_name =  strcat(strcat('../data/pixel_locations', num2str(i)), '.mat');
   img = imread(img_name);
   max_size = 512;
   if (size(img, 1) > max_size || size(img, 2) > max_size)
       img = imresize(img, min(max_size / size(img, 1), max_size / size(img, 2)));
   end
%    img = imresize(img, [512, 512]);
   num_sec = 10;
   num_frames_per_movement = 3;
   num_mosquitos = round(rand(1) * 5) + 1;
   radii = round(rand(1, num_mosquitos) * 1) + 2;
   move_sizes = round(rand(1, num_mosquitos) * 6) + 3;
   generate_mosquito_data(img, num_sec, num_frames_per_movement, num_mosquitos, ...
                          radii, move_sizes, video_name, label_name, pixel_pos_name);
   fprintf('Finished with %d image\n', i)
end