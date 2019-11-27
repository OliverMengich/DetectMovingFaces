faceDetector = vision.CascadeObjectDetector();

 % read video frame ab run the face detector
videoFileReader = vision.VideoFileReader('facemoved.mp4');
videoFrame = step(videoFileReader);
bbox = step(faceDetector,videoFrame);

% draw the returned bounding box around the detected face
videoFrame = insertShape(videoFrame,'Rectangle',bbox);
figure;imshow(videoFrame); title('Detected Face');

% now to convert the first box into a list of 4 points
% this is needed to be able to visualize the rotation of the object

bboxPoints = bbox2points(bbox(1,:));

% now to use the KLT algorithm to set feature points across the videoframe

points = detectMinEigenFeatures(rgb2gray(videoFrame),'ROI',bbox);

% now display the detected points
figure; imshow(videoFrame),hold on, title('Detected Features');
plot(points);
hold off;
% now to initialize a tracker to track the obtained points from the
% detected regions

pointTracker = vision.PointTracker('MaxBidirectionalError',2);

% Initialize the tracker with the initial points locations and the initial
% video frame

points = points.Location;
initialize(pointTracker,points,videoFrame);
% now we create a video player object for displaying video frame

videoPlayer = vision.VideoPlayer('Position',[100 100 [size(videoFrame,2), size(videoFrame,1)]+30]);

% now to track the face

oldpoints = points;

while ~isDone(videoFileReader)
    % get the next frame
    videoFrame = step(videoFileReader);
    
    % Track the points. The old points might be lost in this process
    [points, isFound] = step(pointTracker,videoFrame);
    visiblePoints = points(isFound,:);
    oldInliners = oldpoints(isFound,:);
    
    if size(visiblePoints,1) >=2
        
        % Estimate th geometric transformation between the old points
        % and the new points and eliminate outliners
        
        [xForm,oldInLiners,visiblePoints] = estimateGeometricTransform(oldInliners,...
            visiblePoints,'similarity','MaxDistance',4);
        
        % now apply the transformation 
        bboxPoints = transformPointsForward(xForm,bboxPoints);
        
        % insert a bounding box around the object being tracked
        bboxPolygon = reshape(bboxPoints',1,[]);
        videoFrame = insertShape(videoFrame,'Polygon',bboxPolygon,...
            'LineWidth',2);
        % display tracked points
        videoFrame = insertMarker(videoFrame,visiblePoints,'+','Color','white');
        
        % Reset the points
        oldPoints = visiblePoints;
        setPoints(pointTracker,oldPoints);
        
        
    end
    
    % display the annotated video frame using the video player object
    step(videoPlayer,videoFrame);
end
% cleam up
release(videoFileReader);
release(videoPlayer);
