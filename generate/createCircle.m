function [grid] = createCircle(image_size, position, radius)
    % adjust the position to reflect that position is the top left
%     position = position + [radius, radius];
    [columnsInImage, rowsInImage] = meshgrid(1:image_size(2), 1:image_size(1));
    circlePixels = (rowsInImage - position(2)).^2 + (columnsInImage - position(1)).^2 <= radius.^2;
    grid = circlePixels;