#include "polymetis/clients/franka_hand_client.hpp"

#include "spdlog/spdlog.h"
#include <string>
#include <thread>
#include <time.h>
#include <cmath>
#include <chrono>
#include <limits>
#include <stdexcept>

#include <grpc/grpc.h>

#include "polymetis.grpc.pb.h"
#include "polymetis/utils.h"

using grpc::ClientContext;

FrankaHandClient::FrankaHandClient(std::shared_ptr<grpc::Channel> channel,
                                   YAML::Node config)
    : stub_(GripperServer::NewStub(channel)) {
  // Connect to gripper
  std::string robot_ip = config["robot_ip"].as<std::string>();
  spdlog::info("Connecting to robot_ip {}", robot_ip);
  gripper_.reset(new franka::Gripper(robot_ip));

  // Desk may already have initialized the hand; homing on every launch can fail.
  if (config["home_on_start"] && config["home_on_start"].as<bool>()) {
    spdlog::info("Homing Franka Hand on startup");
    gripper_->homing();
  }
  franka::GripperState franka_gripper_state = gripper_->readOnce();
  if (!std::isfinite(franka_gripper_state.max_width) ||
      franka_gripper_state.max_width < 0.01) {
    throw std::runtime_error(
        "Franka Hand is not initialized (max_width < 0.01 m). "
        "Initialize Franka Hand in Desk, then restart the gripper client.");
  }
  spdlog::info("Franka Hand ready: width={} m, max_width={} m",
               franka_gripper_state.width, franka_gripper_state.max_width);
  if (config["command_deadband_m"]) {
    command_deadband_m_ = config["command_deadband_m"].as<double>();
  }
  if (!std::isfinite(command_deadband_m_) || command_deadband_m_ < 0.0) {
    throw std::invalid_argument("command_deadband_m must be nonnegative");
  }

  // Initialize server connection
  GripperMetadata metadata;
  metadata.set_max_width(franka_gripper_state.max_width);
  metadata.set_hz(GRIPPER_HZ);

  ClientContext context;
  Empty empty;
  stub_->InitRobotClient(&context, metadata, &empty);

  spdlog::info("Connected.", robot_ip);
}

void FrankaHandClient::getGripperState(void) {
  franka::GripperState franka_gripper_state = gripper_->readOnce();

  gripper_state_.set_width(franka_gripper_state.width);
  gripper_state_.set_is_grasped(franka_gripper_state.is_grasped);
  gripper_state_.set_is_moving(is_moving_.load());
  gripper_state_.set_prev_command_successful(prev_cmd_successful_.load());
  gripper_state_.set_device_sample_time_ms(franka_gripper_state.time.toMSec());

  // gripper_state.time();  // Use current timestamp instead!
  setTimestampToNow(gripper_state_.mutable_timestamp());
}

void FrankaHandClient::applyGripperCommand(GripperCommand command) {
  is_moving_.store(true);

  try {
    bool success;
    if (command.grasp()) {
      double eps_inner = command.epsilon_inner() >= 0.0
                             ? command.epsilon_inner() : 0.005;
      double eps_outer = command.epsilon_outer() >= 0.0
                             ? command.epsilon_outer() : 0.005;
      success = gripper_->grasp(command.width(), command.speed(),
                                command.force(), eps_inner, eps_outer);
    } else {
      success = gripper_->move(command.width(), command.speed());
    }
    prev_cmd_successful_.store(success);
  } catch (const std::exception &ex) {
    prev_cmd_successful_.store(false);
    spdlog::error("Franka Hand command failed: {}", ex.what());
  }
  is_moving_.store(false);
}

void FrankaHandClient::run(void) {
  const auto period = std::chrono::nanoseconds(1000000000LL / GRIPPER_HZ);
  double prev_cmd_width = std::numeric_limits<double>::quiet_NaN();
  std::thread movement_thread;
  GripperCommand active_command;
  double active_start_width = std::numeric_limits<double>::quiet_NaN();
  bool stop_requested = false;
  auto next_stop_attempt = std::chrono::steady_clock::time_point::min();
  auto next_wakeup = std::chrono::steady_clock::now();
  while (true) {
    // Run control step
    getGripperState();

    grpc::ClientContext context;
    status_ = stub_->ControlUpdate(&context, gripper_state_, &gripper_cmd_);
    if (!status_.ok()) {
      if (movement_thread.joinable()) movement_thread.join();
      throw std::runtime_error("Gripper ControlUpdate failed: " +
                               status_.error_message());
    }

    const int64_t timestamp_ns =
        static_cast<int64_t>(gripper_cmd_.timestamp().seconds()) * 1000000000LL +
        gripper_cmd_.timestamp().nanos();
    if (is_moving_.load() && !stop_requested &&
        std::chrono::steady_clock::now() >= next_stop_attempt &&
        timestamp_ns > 0 &&
        timestamp_ns != prev_cmd_timestamp_ns_ &&
        std::isfinite(gripper_cmd_.width()) &&
        std::isfinite(gripper_cmd_.speed()) && gripper_cmd_.speed() > 0.0 &&
        gripper_cmd_.width() > gripper_state_.width() + command_deadband_m_ &&
        (active_command.grasp() ||
         active_command.width() < active_start_width - command_deadband_m_)) {
      // A release must not wait for a closing move or grasp to finish.
      next_stop_attempt = std::chrono::steady_clock::now() +
                          std::chrono::milliseconds(250);
      try {
        stop_requested = gripper_->stop();
        if (!stop_requested) {
          spdlog::warn("Could not stop closing Franka Hand command");
        }
      } catch (const std::exception &ex) {
        spdlog::error("Could not stop closing Franka Hand command: {}", ex.what());
      }
    }

    if (!is_moving_.load()) {
      if (movement_thread.joinable()) movement_thread.join();
      const double width = gripper_cmd_.width();
      if (timestamp_ns > 0 && timestamp_ns != prev_cmd_timestamp_ns_ &&
          std::isfinite(width) && std::isfinite(gripper_cmd_.speed()) &&
          gripper_cmd_.speed() > 0.0 &&
          (gripper_cmd_.grasp() || !std::isfinite(prev_cmd_width) ||
           std::abs(width - prev_cmd_width) >= command_deadband_m_)) {
        prev_cmd_timestamp_ns_ = timestamp_ns;
        prev_cmd_width = width;
        active_command = gripper_cmd_;
        active_start_width = gripper_state_.width();
        stop_requested = false;
        is_moving_.store(true);
        movement_thread = std::thread(&FrankaHandClient::applyGripperCommand,
                                      this, gripper_cmd_);
      }
    }

    next_wakeup += period;
    std::this_thread::sleep_until(next_wakeup);
  }
}

int main(int argc, char *argv[]) {
  if (argc != 2) {
    spdlog::error("Usage: franka_hand_client /path/to/cfg.yaml");
    return 1;
  }
  YAML::Node config = YAML::LoadFile(argv[1]);

  // Launch client
  std::string control_address = config["control_ip"].as<std::string>() + ":" +
                                config["control_port"].as<std::string>();
  try {
    FrankaHandClient franka_hand_client(
        grpc::CreateChannel(control_address, grpc::InsecureChannelCredentials()),
        config);
    franka_hand_client.run();
  } catch (const std::exception &ex) {
    spdlog::error("Franka Hand client failed: {}", ex.what());
    return 1;
  }

  // Termination
  spdlog::info("Wait for shutdown; press CTRL+C to close.");

  return 0;
}
