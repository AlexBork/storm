#include "EliminateAutomaticallyAction.h"
#include "EliminateAction.h"

namespace storm {
    namespace jani {
        namespace elimination_actions {
            EliminateAutomaticallyAction::EliminateAutomaticallyAction(const std::string &automatonName, EliminateAutomaticallyAction::EliminationOrder order, uint32_t transitionCountThreshold)
                    : automatonName(automatonName),
                      eliminationOrder(order),
                      transitionCountThreshold(transitionCountThreshold){
            }

            std::string EliminateAutomaticallyAction::getDescription() {
                return "EliminateAutomaticallyAction";
            }

            void EliminateAutomaticallyAction::doAction(JaniLocalEliminator::Session &session) {
                Automaton* automaton = &session.getModel().getAutomaton(automatonName);
                switch (eliminationOrder){
                    case EliminationOrder::Arbitrary: {
                        for (const auto& loc : automaton->getLocations()) {
                            if (session.isEliminable(automatonName, loc.getName())) {
                                session.addToLog("Eliminating location " + loc.getName());
                                EliminateAction action = EliminateAction(automatonName, loc.getName());
                                action.doAction(session);
                            }
                        }
                        break;
                    }
                    case EliminationOrder::NewTransitionCount: {

                        // A location is uneliminable if
                        // - it is initial
                        // - it (potentially) satisfies the property
                        // - it has already been eliminated
                        // - it has a loop
                        // After each elimination, this list is updated to account for new loops
                        std::map<std::string, bool> uneliminable;
                        session.addToLog("Elimination status of locations:");
                        for (const auto& loc : automaton->getLocations()) {
                            bool possiblyInitial = session.isPossiblyInitial(automatonName, loc.getName());
                            bool partOfProp = session.isPartOfProp(automatonName, loc.getName());
                            bool hasLoops = session.hasLoops(automatonName, loc.getName());
                            bool isDeadlock = automaton->getEdgesFromLocation(loc.getName()).empty();

                            bool isUneliminable = possiblyInitial || partOfProp || hasLoops || isDeadlock;
                            uneliminable.insert(std::pair<std::string, bool>(loc.getName(), isUneliminable));

                            std::string eliminationStatus = "\t" + loc.getName() + ": ";
                            if (isUneliminable){
                                eliminationStatus += "Uneliminable (";
                                if (possiblyInitial)
                                    eliminationStatus += "initial, ";
                                if (partOfProp)
                                    eliminationStatus += "part of prop, ";
                                if (hasLoops)
                                    eliminationStatus += "has loops, ";
                                if (isDeadlock)
                                    eliminationStatus += "has no outgoing edges, ";
                                eliminationStatus = eliminationStatus.substr(0, eliminationStatus.size() - 2) + ")";
                            }else{
                                eliminationStatus += "Eliminable";
                            }
                            session.addToLog(eliminationStatus);
                        }

                        session.addToLog("Performing elimination");

                        bool done = false;
                        while (!done) {
                            uint64_t minNewEdges = LLONG_MAX; // TODO: Use actual uint64_t max, as long long isn't guaranteed to be 64 bits
                            int bestLocIndex = -1;
                            for (const auto& loc : automaton->getLocations()) {
                                if (uneliminable[loc.getName()])
                                    continue;

                                int locIndex = automaton->getLocationIndex(loc.getName());
                                uint64_t outgoing = automaton->getEdgesFromLocation(locIndex).size();
                                uint64_t incoming = 0;
                                for (const auto& edge : automaton->getEdges()) {
                                    int addedTransitions = 1;
                                    for (const auto& dest : edge.getDestinations())
                                        if (dest.getLocationIndex() == locIndex){
                                            addedTransitions *= outgoing;
                                            // Stop once we hit the threshold -- otherwise there is a risk of causing
                                            // an overflow due to the exponential growth of addedTransitions:
                                            if (addedTransitions > transitionCountThreshold){
                                                break;
                                            }
                                        }
                                    incoming += addedTransitions - 1;
                                }
                                uint64_t newEdges = incoming * outgoing;

                                if (newEdges <= minNewEdges){
                                    minNewEdges = newEdges;
                                    bestLocIndex = locIndex;
                                }
                            }


                            if (bestLocIndex == -1){
                                done = true;
                                session.addToLog("Cannot eliminate more locations");
                            } else if (minNewEdges > transitionCountThreshold) {
                                done = true;
                                session.addToLog(
                                        "Cannot eliminate more locations without creating too many new transitions (best: >" +
                                        std::to_string(minNewEdges) + " new transitions)");
                            } else {
                                std::string locName = automaton->getLocation(bestLocIndex).getName();
                                session.addToLog("\tEliminating location " + locName + " (" + std::to_string(minNewEdges) + " new edges)");
                                EliminateAction action = EliminateAction(automatonName, locName);
                                action.doAction(session);
                                automaton = &session.getModel().getAutomaton(automatonName);
                                uneliminable[locName] = true;

                                // Update "uneliminable" to account for potential new loops
                                for (const auto& loc : automaton->getLocations()) {
                                    if (!uneliminable[loc.getName()]){
                                        if (session.hasLoops(automatonName, loc.getName())){
                                            uneliminable[loc.getName()] = true;
                                            session.addToLog("\t" + loc.getName() + " now has a loop");
                                        }
                                    }
                                }
                            }
                        }

                        break;
                    }
                    default: {
                        STORM_LOG_THROW(true, storm::exceptions::NotImplementedException, "This elimination order is not yet implemented");
                        break;
                    }
                }
            }

            std::string EliminateAutomaticallyAction::find_next_location(JaniLocalEliminator::Session &session) {
                return "";
            }
        }
    }
}