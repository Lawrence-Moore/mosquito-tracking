function [grid, blendedImage] = insertMosquito(image, position, pixelSize)
    fullPixels = floor(pixelSize);
    grid = zeros(size(image));
    blendedImage = image;

    for x=position(1) - fullPixels/2: position(1) + fullPixels/2 - 1
        for y=position(2) - fullPixels/2:position(2) + fullPixels/2 - 1
            % bound check
            xindex = min(max(1, x), size(image, 1));
            yindex = min(max(1, y), size(image, 2));
            blendedImage(xindex, yindex, :) = 0;
            grid(xindex, yindex) = 1;
        end
    end
    
    % darken the surrounding pixels
    darkenFactor = 2;
    if (pixelSize > fullPixels)
        for offset= -ceil(pixelSize/2) : ceil(pixelSize/2) - 1
            % bound check
            pos1 = [position(1) - ceil(pixelSize/2), position(2) + offset];
            pos2 = [position(1) + ceil(pixelSize/2) - 1, position(2) + offset];
            pos3 = [position(1) + offset, position(2) - ceil(pixelSize/2)];
            pos4 = [position(1) + offset, position(2) + ceil(pixelSize/2) - 1];
            if (checkBounds(image, pos1(1), pos1(2)))
                blendedImage(pos1(1), pos1(2), :) = image(pos1(1), pos1(2), :) / darkenFactor;
                grid(pos1(1), pos1(2)) = 1;
            end
            if (checkBounds(image, pos2(1, pos2(2))))
                blendedImage(pos2(1), pos2(2), :) = image(pos2(1), pos2(2), :) / darkenFactor;
                grid(pos2(1), pos2(2)) = 1;
            end
            if (checkBounds(image, pos3(1, pos3(2))))
                blendedImage(pos3(1), pos3(2), :) = image(pos3(1), pos3(2), :) / darkenFactor;
                grid(pos3(1), pos3(2)) = 1;
            end
            if (checkBounds(image, pos4(1, pos4(2))))
                blendedImage(pos4(1), pos4(2), :) = image(pos4(1), pos4(2), :) / darkenFactor;
                grid(pos3(1), pos3(2)) = 1;
            end
        end
    end
end

function [withinB] = checkBounds(image, x, y)
    withinB = ~(x < 1 || x > size(image , 1) || y < 1 || y > size(image, 1));
end