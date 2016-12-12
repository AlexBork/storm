#include "storm/storage/jani/TemplateEdge.h"

#include "storm/storage/jani/Model.h"

namespace storm {
    namespace jani {
        
        TemplateEdge::TemplateEdge(storm::expressions::Expression const& guard) : guard(guard) {
            // Intentionally left empty.
        }
        
        void TemplateEdge::addDestination(TemplateEdgeDestination const& destination) {
            destinations.emplace_back(destination);
        }
        
        bool TemplateEdge::addTransientAssignment(Assignment const& assignment) {
            return assignments.add(assignment);
        }
        
        void TemplateEdge::finalize(Model const& containingModel) {
            for (auto const& destination : getDestinations()) {
                for (auto const& assignment : destination.getOrderedAssignments().getAllAssignments()) {
                    if (containingModel.getGlobalVariables().hasVariable(assignment.getExpressionVariable())) {
                        writtenGlobalVariables.insert(assignment.getExpressionVariable());
                    }
                }
            }
        }
        
        boost::container::flat_set<storm::expressions::Variable> const& TemplateEdge::getWrittenGlobalVariables() const {
            return writtenGlobalVariables;
        }
        
        storm::expressions::Expression const& TemplateEdge::getGuard() const {
            return guard;
        }
        
        std::size_t TemplateEdge::getNumberOfDestinations() const {
            return destinations.size();
        }
        
        std::vector<TemplateEdgeDestination> const& TemplateEdge::getDestinations() const {
            return destinations;
        }
        
        TemplateEdgeDestination const& TemplateEdge::getDestination(uint64_t index) const {
            return destinations[index];
        }

        OrderedAssignments const& TemplateEdge::getAssignments() const {
            return assignments;
        }
        
        void TemplateEdge::substitute(std::map<storm::expressions::Variable, storm::expressions::Expression> const& substitution) {
            guard = guard.substitute(substitution);

            for (auto& assignment : assignments) {
                assignment.substitute(substitution);
            }

            for (auto& destination : destinations) {
                destination.substitute(substitution);
            }
        }
        
        void TemplateEdge::changeAssignmentVariables(std::map<Variable const*, std::reference_wrapper<Variable const>> const& remapping) {
            for (auto& destination : destinations) {
                destination.changeAssignmentVariables(remapping);
            }
            assignments.changeAssignmentVariables(remapping);
        }

        void TemplateEdge::liftTransientDestinationAssignments() {
            if (!destinations.empty()) {
                auto const& destination = *destinations.begin();
                
                for (auto const& assignment : destination.getOrderedAssignments().getTransientAssignments()) {
                    // Check if we can lift the assignment to the edge.
                    bool canBeLifted = true;
                    for (auto const& destination : destinations) {
                        if (!destination.hasAssignment(assignment)) {
                            canBeLifted = false;
                            break;
                        }
                    }
                    
                    // If so, remove the assignment from all destinations.
                    if (canBeLifted) {
                        this->addTransientAssignment(assignment);
                        for (auto& destination : destinations) {
                            destination.removeAssignment(assignment);
                        }
                    }
                }
            }
        }
        
        void TemplateEdge::pushAssignmentsToDestinations() {
            STORM_LOG_ASSERT(!destinations.empty(), "Need non-empty destinations for this transformation.");
            for (auto const& assignment : this->getAssignments()) {
                for (auto& destination : destinations) {
                    destination.addAssignment(assignment);
                }
            }
            this->assignments.clear();
        }
        
        bool TemplateEdge::usesVariablesInNonTransientAssignments(std::set<storm::expressions::Variable> const& variables) const {
            for (auto const& destination : destinations) {
                for (auto const& assignment : destination.getOrderedAssignments().getNonTransientAssignments()) {
                    if (assignment.getAssignedExpression().containsVariable(variables)) {
                        return true;
                    }
                }
            }
            return false;
        }
        
        bool TemplateEdge::hasTransientEdgeDestinationAssignments() const {
            for (auto const& destination : this->getDestinations()) {
                if (destination.hasTransientAssignment()) {
                    return true;
                }
            }
            return false;
        }
        
        bool TemplateEdge::usesAssignmentLevels() const {
            for (auto const& destination : this->getDestinations()) {
                if (destination.usesAssignmentLevels()) {
                    return true;
                }
            }
            return false;
        }
        
    }
}
