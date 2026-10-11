cmake_minimum_required(VERSION 3.8)
project(crater_sim)

find_package(ament_cmake REQUIRED)

install(PROGRAMS
  tools/gen_world.py
  tools/meteor_spawner.py
  tools/eval_meteors.py
  tools/collect_dataset.py
  tools/make_dataset.sh
  tools/sensor_viz.py
  tools/train_yolo.sh
  tools/gen_rocks.py
  tools/dataset_preview.py
  tools/eval_perception.py
  DESTINATION lib/${PROJECT_NAME})

install(DIRECTORY launch config worlds models gz_systems
  DESTINATION share/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_cmake_pytest REQUIRED)
  ament_add_pytest_test(test_gen_rocks test/test_gen_rocks.py)
endif()

ament_package()
