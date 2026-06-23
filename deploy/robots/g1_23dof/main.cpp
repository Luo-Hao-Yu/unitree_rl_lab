#include "FSM/CtrlFSM.h"
#include "FSM/State_Passive.h"
#include "FSM/State_FixStand.h"
#include "FSM/State_RLBase.h"
#include <chrono>
#include <vector>

std::unique_ptr<LowCmd_t> FSMState::lowcmd = nullptr;
std::shared_ptr<LowState_t> FSMState::lowstate = nullptr;
std::shared_ptr<Keyboard> FSMState::keyboard = nullptr;

void init_fsm_state()
{
    auto lowcmd_sub = std::make_shared<unitree::robot::g1::subscription::LowCmd>();
    usleep(0.2 * 1e6);
    if(!lowcmd_sub->isTimeout())
    {
        spdlog::critical("The other process is using the lowcmd channel, please close it first.");
        // unitree::robot::g1::shutdown();  // not available in current g1 wrapper
        // exit(0);
    }
    FSMState::lowcmd = std::make_unique<LowCmd_t>();
    FSMState::lowstate = std::make_shared<LowState_t>();
    spdlog::info("Waiting for connection to robot...");
    FSMState::lowstate->wait_for_connection();
    spdlog::info("Connected to robot.");
}

int main(int argc, char** argv)
{
    // Load parameters
    auto vm = param::helper(argc, argv);

    std::cout << " --- Unitree Robotics --- \n";
    std::cout << "     G1-23dof Controller \n";

    // Unitree DDS Config
    unitree::robot::ChannelFactory::Instance()->Init(1, vm["network"].as<std::string>());

    init_fsm_state();

    FSMState::lowcmd->msg_.mode_machine() = 4; // 23dof
    if(!FSMState::lowcmd->check_mode_machine(FSMState::lowstate)) {
        spdlog::critical("Unmatched robot type.");
        exit(-1);
    }
    
    // Initialize FSM
    auto fsm = std::make_unique<CtrlFSM>(param::config["FSM"]);

    // ------------------------------------------------------------
    // Auto FSM transition for cloud server deployment without joystick.
    // Timeline:
    //   t > AUTO_FIXSTAND_AFTER : Passive  -> FixStand
    //   t > AUTO_POLICY_AFTER   : FixStand -> RLBase / policy state
    // ------------------------------------------------------------
    const double AUTO_FIXSTAND_AFTER = 1e9;
    const double AUTO_POLICY_AFTER = 1e9;

    auto start_time = std::make_shared<std::chrono::steady_clock::time_point>(
        std::chrono::steady_clock::now()
    );

    auto elapsed_sec = [start_time]() -> double {
        return std::chrono::duration<double>(
            std::chrono::steady_clock::now() - *start_time
        ).count();
    };

    auto find_state_id = [&](const std::vector<std::string>& preferred,
                             const std::vector<std::string>& excluded) -> int {
        for (const auto& name : preferred) {
            if (FSMStringMap.right.count(name)) {
                return FSMStringMap.right.at(name);
            }
        }

        for (auto& s : fsm->states) {
            std::string name = s->getStateString();
            bool skip = false;
            for (const auto& ex : excluded) {
                if (name == ex) {
                    skip = true;
                    break;
                }
            }
            if (!skip) {
                return s->getState();
            }
        }

        spdlog::critical("Auto FSM: failed to find target state.");
        std::exit(-1);
    };

    const int fixstand_id = find_state_id(
        {"FixStand"},
        {"Passive"}
    );

    const int policy_id = find_state_id(
        {"RLBase", "RL", "Policy", "Locomotion", "Velocity"},
        {"Passive", "FixStand"}
    );

    for (auto& s : fsm->states) {
        const std::string name = s->getStateString();

        if (name == "Passive") {
            s->registered_checks.push_back({
                [elapsed_sec, AUTO_FIXSTAND_AFTER]() -> bool {
                    return elapsed_sec() > AUTO_FIXSTAND_AFTER;
                },
                fixstand_id
            });
            spdlog::info(
                "Auto FSM: Passive -> FixStand after {:.1f}s",
                AUTO_FIXSTAND_AFTER
            );
        }

        if (name == "FixStand") {
            s->registered_checks.push_back({
                [elapsed_sec, AUTO_POLICY_AFTER]() -> bool {
                    return elapsed_sec() > AUTO_POLICY_AFTER;
                },
                policy_id
            });
            spdlog::info(
                "Auto FSM: FixStand -> policy state after {:.1f}s",
                AUTO_POLICY_AFTER
            );
        }
    }

    fsm->start();

    std::cout << "Auto FSM disabled: use joystick to switch FSM.\n";
    std::cout << "Joystick is required: LT + Up -> FixStand, RB + X -> Velocity.\n";

    while (true)
    {
        sleep(1);
    }
    
    return 0;
}

