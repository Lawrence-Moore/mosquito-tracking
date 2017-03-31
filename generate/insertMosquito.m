function [grid, blendedImage] = insertMosquito(image, positions, radii)
    grid = zeros(size(image, 1), size(image, 2));
    blendedImage = image;
    for i=1:size(positions, 1)
        position = positions(i, :);
        fullPixels = floor(radii(i));

        mosquito = createCircle(size(image), position, fullPixels);
        
        darken_mask = mosquito .* rand(size(mosquito));
        darken_mask(~mosquito) = 1;
%         darken_mask = rand(size(mosquito));
        
        blendedImage = uint8(double(blendedImage) .* darken_mask);
%         blendedImage = blendedImage .* uint8(repmat(~mosquito, [1,1,3]));
        grid(mosquito) = 1;
    end
    blendedImage = imnoise(blendedImage, 'gaussian', 0, 0.00005);
    
    grid = cat(3, grid, grid, grid); 
end

function [withinB] = checkBounds(image, x, y)
    withinB = ~(x < 1 || x > size(image , 1) || y < 1 || y > size(image, 1));
end