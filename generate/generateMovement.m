function [newPositions, newTrajs] = generateMovement(oldPositions, oldTrajs, moveSizes, imageSize, radii)

newPositions = zeros(size(oldPositions));
newTrajs = zeros(size(oldTrajs));

for i=1:size(oldTrajs, 1)
    oldTraj = oldTrajs(i, :);
    oldPosition = oldPositions(i, :);
    moveSize = moveSizes(i);

    % chance of the mosquito staying course
    same_direction_chance = 0.30;

    % chance of the moquito not moving 
    no_move_chance = 0.05;

    chance = rand();

    if (chance <= same_direction_chance)
        % generate an angle within 30 degrees of the original direction
        angle = -1 * pi / 12 + rand() * pi / 6;

    elseif (chance <= 1.0 - no_move_chance)
        % generate an angle between 15 and 165
        angle = pi / 12 + rand() * 5 * pi / 6;
    else
        angle = 0;
    end

    % compute rotated coordinates
    newTraj = [cos(angle) * oldTraj(1) - sin(angle) * oldTraj(2),...
               sin(angle) * oldTraj(1) + cos(angle) * oldTraj(2)];

    newTraj = newTraj/ sum(abs(newTraj)) * moveSize;

    % check bounds. if out, reverse direction
    newPosition = round(oldPosition  + newTraj);
    if (newPosition(1) + radii(i) > imageSize(2) || newPosition(1)-radii(i)< 0)
        newTraj(1) = -1 * newTraj(1);
    end
    if (newPosition(2) + radii(i) > imageSize(1) || newPosition(2)-radii(i)< 0)
        newTraj(2) = -1 * newTraj(2);
    end

    % update the new position if necessary
    newPosition = floor(oldPosition  + newTraj);
    
    newPositions(i, :) = newPosition;
    newTrajs(i, :) = newTraj;
end
end