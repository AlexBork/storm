#include "src/storage/jani/Variable.h"

#include "src/storage/jani/BooleanVariable.h"
#include "src/storage/jani/BoundedIntegerVariable.h"
#include "src/storage/jani/UnboundedIntegerVariable.h"
#include "src/storage/jani/RealVariable.h"

namespace storm {
    namespace jani {

        Variable::Variable(std::string const& name, storm::expressions::Variable const& variable, storm::expressions::Expression const& init, bool transient) : name(name), variable(variable),  transient(transient), init(init) {
            // Intentionally left empty.
        }

        Variable::Variable(std::string const& name, storm::expressions::Variable const& variable, bool transient) : name(name), variable(variable), transient(transient), init() {
            // Intentionally left empty.
        }


        storm::expressions::Variable const& Variable::getExpressionVariable() const {
            return variable;
        }

        std::string const& Variable::getName() const {
            return name;
        }
        
        bool Variable::isBooleanVariable() const {
            return false;
        }
        
        bool Variable::isBoundedIntegerVariable() const {
            return false;
        }
        
        bool Variable::isUnboundedIntegerVariable() const {
            return false;
        }

        bool Variable::isRealVariable() const {
            return false;
        }
        
        bool Variable::isTransientVariable() const {
            return transient;
        }

        bool Variable::hasInitExpression() const {
            return static_cast<bool>(init);
        }

        storm::expressions::Expression const& Variable::getInitExpression() const {
            return this->init.get();
        }
        
        void Variable::setInitExpression(storm::expressions::Expression const& initialExpression) {
            this->init = initialExpression;
        }
        
        BooleanVariable& Variable::asBooleanVariable() {
            return static_cast<BooleanVariable&>(*this);
        }
        
        BooleanVariable const& Variable::asBooleanVariable() const {
            return static_cast<BooleanVariable const&>(*this);
        }
        
        BoundedIntegerVariable& Variable::asBoundedIntegerVariable() {
            return static_cast<BoundedIntegerVariable&>(*this);
        }
        
        BoundedIntegerVariable const& Variable::asBoundedIntegerVariable() const {
            return static_cast<BoundedIntegerVariable const&>(*this);
        }
        
        UnboundedIntegerVariable& Variable::asUnboundedIntegerVariable() {
            return static_cast<UnboundedIntegerVariable&>(*this);
        }
        
        UnboundedIntegerVariable const& Variable::asUnboundedIntegerVariable() const {
            return static_cast<UnboundedIntegerVariable const&>(*this);
        }
        
        RealVariable& Variable::asRealVariable() {
            return static_cast<RealVariable&>(*this);
        }
        
        RealVariable const& Variable::asRealVariable() const {
            return static_cast<RealVariable const&>(*this);
        }
        
    }
}