% For our extension, we implemented a data visualization gui, similar to 
% the 251 project. This was done using MatLab Guide. The gui includes a 
% display window that plots a Gaussian distribution, with mean and variance 
% values that can adjusted with sliders and updated with the plot button. 
% We chose this extension because we wanted to further explore gui capabilities 
% in MatLab, because we had not discussed them in class.


function varargout = plotter(varargin)
% PLOTTER MATLAB code for plotter.fig
%      PLOTTER, by itself, creates a new PLOTTER or raises the existing
%      singleton*.
%
%      H = PLOTTER returns the handle to a new PLOTTER or the handle to
%      the existing singleton*.
%
%      PLOTTER('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in PLOTTER.M with the given input arguments.
%
%      PLOTTER('Property','Value',...) creates a new PLOTTER or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before plotter_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to plotter_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help plotter

% Last Modified by GUIDE v2.5 13-Feb-2020 19:07:45

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @plotter_OpeningFcn, ...
                   'gui_OutputFcn',  @plotter_OutputFcn, ...
                   'gui_LayoutFcn',  [] , ...
                   'gui_Callback',   []);
if nargin && ischar(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end
% End initialization code - DO NOT EDIT


% --- Executes just before plotter is made visible.
function plotter_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to plotter (see VARARGIN)

% define function to make gaussian distribution
handles.gauss = @(x, mu, sigmasqr) 1/(sqrt(2*pi*sigmasqr)) * exp(-((x-mu).^2)/sigmasqr);

% start with a standard normal distribution
handles.mu = 0;
handles.var = 1;
x = [-3*sqrt(handles.var)+handles.mu-0.1:.025:3*sqrt(handles.var)+handles.mu+0.1];
y = handles.gauss(x, handles.mu, handles.var);
plot(x,y);
    
% Choose default command line output for plotter
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);

% UIWAIT makes plotter wait for user response (see UIRESUME)
% uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = plotter_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;


% --- Executes on button press in pushbutton1.
function pushbutton1_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Make a standard normal distribution with the mean and the variance
% as described by the values in the sliders
mu = get(handles.slider1, "Value");
var = get(handles.slider2, "Value");
x = [-3*sqrt(var)+mu-0.1:.025:3*sqrt(var)+mu+0.1];
y = handles.gauss(x, mu, var);
plot(x,y);




% --- Executes on slider movement.
function slider1_Callback(hObject, eventdata, handles)
% hObject    handle to slider1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% Hints: get(hObject,'Value') returns position of slider
%        get(hObject,'Min') and get(hObject,'Max') to determine range of slider


% --- Executes during object creation, after setting all properties.
function slider1_CreateFcn(hObject, eventdata, handles)
% hObject    handle to slider1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called
handles.slider1 = hObject;
% Hint: slider controls usually have a light gray background.
if isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor',[.9 .9 .9]);
end


% --- Executes on slider movement.
function slider2_Callback(hObject, eventdata, handles)
% hObject    handle to slider2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% Hints: get(hObject,'Value') returns position of slider
%        get(hObject,'Min') and get(hObject,'Max') to determine range of slider


% --- Executes during object creation, after setting all properties.
function slider2_CreateFcn(hObject, eventdata, handles)
% hObject    handle to slider2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called
handles.slider2 = hObject;
% Hint: slider controls usually have a light gray background.
if isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor',[.9 .9 .9]);
end
