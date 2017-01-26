function [grid, blendedImage] = insertMosquito(image, positions, radii)
    grid = zeros(size(image, 1), size(image, 2));
    blendedImage = image;
    for i=1:size(positions, 1)
        position = positions(i, :);
        fullPixels = floor(radii(1));

        mosquito = createCircle(size(image), position, fullPixels);
        
        darkenFactor = 20;
        blendedImage = blendedImage .* uint8(repmat(~mosquito, [1,1,3]));
        grid(mosquito) = 1;
    
% 
%         % darken the surrounding pixels
%         darkenFactor = 2;
%         if (pixelSize > fullPixels)
%             for offset= -ceil(pixelSize/2) : ceil(pixelSize/2) - 1
%                 % bound check
%                 pos1 = [position(1) - ceil(pixelSize/2), position(2) + offset];
%                 pos2 = [position(1) + ceil(pixelSize/2) - 1, position(2) + offset];
%                 pos3 = [position(1) + offset, position(2) - ceil(pixelSize/2)];
%                 pos4 = [position(1) + offset, position(2) + ceil(pixelSize/2) - 1];
%                 if (checkBounds(image, pos1(1), pos1(2)))
%                     blendedImage(pos1(1), pos1(2), :) = image(pos1(1), pos1(2), :) / darkenFactor;
%                     grid(pos1(1), pos1(2)) = 1;
%                 end
%                 if (checkBounds(image, pos2(1, pos2(2))))
%                     blendedImage(pos2(1), pos2(2), :) = image(pos2(1), pos2(2), :) / darkenFactor;
%                     grid(pos2(1), pos2(2)) = 1;
%                 end
%                 if (checkBounds(image, pos3(1, pos3(2))))
%                     blendedImage(pos3(1), pos3(2), :) = image(pos3(1), pos3(2), :) / darkenFactor;
%                     grid(pos3(1), pos3(2)) = 1;
%                 end
%                 if (checkBounds(image, pos4(1, pos4(2))))
%                     blendedImage(pos4(1), pos4(2), :) = image(pos4(1), pos4(2), :) / darkenFactor;
%                     grid(pos3(1), pos3(2)) = 1;
%                 end
%             end
%         end
    end
    
    grid = cat(3, grid, grid, grid); 
end

function [withinB] = checkBounds(image, x, y)
    withinB = ~(x < 1 || x > size(image , 1) || y < 1 || y > size(image, 1));
end