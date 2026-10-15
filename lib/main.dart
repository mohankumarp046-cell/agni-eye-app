import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'package:flutter_vision/flutter_vision.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  final cameras = await availableCameras();
  runApp(MyApp(camera: cameras.first));
}

class MyApp extends StatelessWidget {
  final CameraDescription camera;
  const MyApp({Key? key, required this.camera}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      theme: ThemeData.dark(),
      title: 'AGNI EYE - Crack Detection',
      home: DetectionScreen(camera: camera),
    );
  }
}

class DetectionScreen extends StatefulWidget {
  final CameraDescription camera;
  const DetectionScreen({Key? key, required this.camera}) : super(key: key);

  @override
  _DetectionScreenState createState() => _DetectionScreenState();
}

class _DetectionScreenState extends State<DetectionScreen> {
  late CameraController _controller;
  late FlutterVision vision;
  bool isLoaded = false;
  bool isDetecting = false;
  List<Map<String, dynamic>> yoloResults = [];
  int imageHeight = 0;
  int imageWidth = 0;

  @override
  void initState() {
    super.initState();
    vision = FlutterVision();
    initCameraAndModel();
  }

  initCameraAndModel() async {
    // 1. Camera setup
    _controller = CameraController(widget.camera, ResolutionPreset.medium);
    await _controller.initialize();

    // 2. YOLOv8 TFLite model load
    await vision.loadYoloModel(
      modelPath: 'assets/best.tflite',
      labels: 'assets/labels.txt',
      imageWidth: 640,
      imageHeight: 640,
      task: "detect",
      numThreads: 4,
      useGpu: true,
    );

    setState(() {
      isLoaded = true;
    });

    // 3. Live camera stream detection
    startDetection();
  }

  void startDetection() {
    _controller.startImageStream((CameraImage image) async {
      if (isDetecting) return;
      setState(() {
        isDetecting = true;
      });

      // Frame-a model-ku pass panrom
      final result = await vision.yoloOnFrame(
        bytesList: image.planes.map((plane) => plane.bytes).toList(),
        imageHeight: image.height,
        imageWidth: image.width,
        iouThreshold: 0.4,
        confThreshold: 0.4,
      );

      if (mounted) {
        setState(() {
          yoloResults = result;
          imageHeight = image.height;
          imageWidth = image.width;
          isDetecting = false;
        });
      }
    });
  }

  @override
  void dispose() {
    _controller.stopImageStream();
    _controller.dispose();
    vision.closeYoloModel();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    if (!isLoaded || !_controller.value.isInitialized) {
      return const Scaffold(
        body: Center(child: CircularProgressIndicator(color: Colors.redAccent)),
      );
    }

    final Size screenSize = MediaQuery.of(context).size;

    return Scaffold(
      appBar: AppBar(
        title: const Text('AGNI EYE - Live Crack Detection'),
        backgroundColor: Colors.red[900],
      ),
      body: Stack(
        children: [
          // 1. Camera Preview
          CameraPreview(_controller),

          // 2. Crack Bounding Box Draw Overlay
          ...yoloResults.map((result) {
            final box = result["box"];
            final String tag = result["tag"] ?? "Crack";
            final double confidence = (result["box"][4] ?? 0.0) * 100;

            // Screen scale ratios
            final double factorX = screenSize.width / imageWidth;
            final double factorY = screenSize.height / imageHeight;

            final double left = box[0] * factorX;
            final double top = box[1] * factorY;
            final double width = (box[2] - box[0]) * factorX;
            final double height = (box[3] - box[1]) * factorY;

            return Positioned(
              left: left,
              top: top,
              width: width,
              height: height,
              child: Container(
                decoration: BoxDecoration(
                  border: Border.all(color: Colors.red, width: 3.0),
                  borderRadius: BorderRadius.circular(4.0),
                ),
                child: Align(
                  alignment: Alignment.topLeft,
                  child: Container(
                    color: Colors.red[900],
                    padding: const EdgeInsets.symmetric(horizontal: 4, vertical: 2),
                    child: Text(
                      '$tag ${confidence.toStringAsFixed(1)}%',
                      style: const TextStyle(
                        color: Colors.white,
                        fontSize: 12,
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                  ),
                ),
              ),
            );
          }).toList(),

          // 3. Status Display Bar
          Positioned(
            bottom: 30,
            left: 20,
            right: 20,
            child: Container(
              padding: const EdgeInsets.all(12),
              decoration: BoxDecoration(
                color: Colors.black87,
                borderRadius: BorderRadius.circular(10),
                border: Border.all(color: Colors.redAccent, width: 1.5),
              ),
              child: Row(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  Icon(
                    yoloResults.isNotEmpty ? Icons.warning_amber_rounded : Icons.check_circle_outline,
                    color: yoloResults.isNotEmpty ? Colors.redAccent : Colors.greenAccent,
                  ),
                  const SizedBox(width: 10),
                  Text(
                    yoloResults.isNotEmpty
                        ? 'CRACK DETECTED (${yoloResults.length})'
                        : 'Status: Surface Normal / Scanning...',
                    style: TextStyle(
                      color: yoloResults.isNotEmpty ? Colors.redAccent : Colors.white,
                      fontSize: 15,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }
}
