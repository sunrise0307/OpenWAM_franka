#include "spdlog/spdlog.h"
#include "yaml-cpp/yaml.h"

#include "polymetis/utils.h"
#include <grpcpp/grpcpp.h>

#include <franka/gripper.h>
#include <franka/gripper_state.h>
#include <atomic>
#include <cstdint>

#define GRIPPER_HZ 30

// Define tolerances to be able to grasp any object without specifying width
#define EPSILON_INNER 0.2
#define EPSILON_OUTER 0.2

class FrankaHandClient {
private:
  void getGripperState(void);
  void applyGripperCommand(GripperCommand command);

  // gRPC
  std::unique_ptr<GripperServer::Stub> stub_;
  grpc::Status status_;

  GripperState gripper_state_;
  GripperCommand gripper_cmd_;
  int64_t prev_cmd_timestamp_ns_ = -1;
  std::atomic<bool> prev_cmd_successful_{true};

  // Franka
  std::shared_ptr<franka::Gripper> gripper_;
  std::atomic<bool> is_moving_{false};
  double command_deadband_m_ = 0.001;

public:
  FrankaHandClient(std::shared_ptr<grpc::Channel> channel, YAML::Node config);
  void run(void);
};
