#include "storm/utility/constants.h"

#include <type_traits>

#include "storm/storage/sparse/StateType.h"
#include "storm/storage/SparseMatrix.h"
#include "storm/settings/SettingsManager.h"
#include "storm/settings/modules/GeneralSettings.h"

#include "storm/exceptions/InvalidArgumentException.h"

#include "storm/adapters/CarlAdapter.h"
#include "storm/utility/macros.h"

namespace storm {
    namespace utility {
        
        template<typename ValueType>
        ValueType one() {
            return ValueType(1);
        }

        template<typename ValueType>
        ValueType zero() {
            return ValueType(0);
        }

        template<typename ValueType>
        ValueType infinity() {
            return std::numeric_limits<ValueType>::infinity();
        }

        template<typename ValueType>
        bool isOne(ValueType const& a) {
            return a == one<ValueType>();
        }

        template<typename ValueType>
        bool isZero(ValueType const& a) {
            return a == zero<ValueType>();
        }

        template<typename ValueType>
        bool isConstant(ValueType const&) {
            return true;
        }
        
        template<typename ValueType>
        bool isInfinity(ValueType const& a) {
            return a == infinity<ValueType>();
        }

        template<typename ValueType>
        bool isInteger(ValueType const& number) {
            ValueType iPart;
            ValueType result = std::modf(number, &iPart);
            return result == zero<ValueType>();
        }
                
        template<>
        bool isInteger(int const&) {
            return true;
        }
        
        template<>
        bool isInteger(uint32_t const&) {
            return true;
        }

        template<>
        bool isInteger(storm::storage::sparse::state_type const&) {
            return true;
        }

        template<>
        uint_fast64_t convertNumber(double const& number){
            return std::llround(number);
        }
        
        template<>
        double convertNumber(uint_fast64_t const& number){
            return number;
        }
        
        template<>
        double convertNumber(double const& number){
            return number;
        }
        
        template<typename ValueType>
        ValueType simplify(ValueType value) {
            // In the general case, we don't do anything here, but merely return the value. If something else is
            // supposed to happen here, the templated function can be specialized for this particular type.
            return value;
        }

        template<typename IndexType, typename ValueType>
        storm::storage::MatrixEntry<IndexType, ValueType> simplify(storm::storage::MatrixEntry<IndexType, ValueType> matrixEntry) {
            simplify(matrixEntry.getValue());
            return matrixEntry;
        }
        
        template<typename IndexType, typename ValueType>
        storm::storage::MatrixEntry<IndexType, ValueType>& simplify(storm::storage::MatrixEntry<IndexType, ValueType>& matrixEntry) {
            simplify(matrixEntry.getValue());
            return matrixEntry;
        }
        
        template<typename IndexType, typename ValueType>
        storm::storage::MatrixEntry<IndexType, ValueType>&& simplify(storm::storage::MatrixEntry<IndexType, ValueType>&& matrixEntry) {
            simplify(matrixEntry.getValue());
            return std::move(matrixEntry);
        }

        template<typename ValueType>
        std::pair<ValueType, ValueType> minmax(std::vector<ValueType> const& values) {
            assert(!values.empty());
            ValueType min = values.front();
            ValueType max = values.front();
            for (auto const& vt : values) {
                if (vt < min) {
                    min = vt;
                }
                if (vt > max) {
                    max = vt;
                }
            }
            return std::make_pair(min, max);
        }
        
        template<typename ValueType>
        ValueType minimum(std::vector<ValueType> const& values) {
            return minmax(values).first;
        }
        
        
        template<typename ValueType>
        ValueType maximum(std::vector<ValueType> const& values) {
            return minmax(values).second;
        }
        
        template<typename K, typename ValueType>
        std::pair<ValueType, ValueType> minmax(std::map<K, ValueType> const& values) {
            assert(!values.empty());
            ValueType min = values.begin()->second;
            ValueType max = values.begin()->second;
            for (auto const& vt : values) {
                if (vt.second < min) {
                    min = vt.second;
                }
                if (vt.second > max) {
                    max = vt.second;
                }
            }
            return std::make_pair(min, max);
        }
        
        template< typename K, typename ValueType>
        ValueType minimum(std::map<K, ValueType> const& values) {
            return minmax(values).first;
        }
        
        template<typename K, typename ValueType>
        ValueType maximum(std::map<K, ValueType> const& values) {
            return minmax(values).second;
        }
        
        template<typename ValueType>
        ValueType pow(ValueType const& value, uint_fast64_t exponent) {
            return std::pow(value, exponent);
        }

        template<typename ValueType>
        ValueType sqrt(ValueType const& number) {
            return std::sqrt(number);
        }
        
        template<typename ValueType>
        ValueType abs(ValueType const& number) {
            return std::fabs(number);
        }
        
        template<typename ValueType>
        ValueType floor(ValueType const& number) {
            return std::floor(number);
        }
        
        template<typename ValueType>
        ValueType ceil(ValueType const& number) {
            return std::ceil(number);
        }

        template<typename ValueType>
        std::string to_string(ValueType const& value) {
            std::stringstream ss;
            ss << value;
            return ss.str();
        }
        
#if defined(STORM_HAVE_CARL) && defined(STORM_HAVE_CLN)
        template<>
        storm::ClnRationalNumber infinity() {
            // FIXME: this should be treated more properly.
            return storm::ClnRationalNumber(-1);
        }

        template<>
        bool isOne(storm::ClnRationalNumber const& a) {
            return carl::isOne(a);
        }

        template<>
        bool isZero(storm::ClnRationalNumber const& a) {
            return carl::isZero(a);
        }
        
        template<>
        bool isInteger(storm::ClnRationalNumber const& number) {
            return carl::isInteger(number);
        }

        template<>
        std::pair<storm::ClnRationalNumber, storm::ClnRationalNumber> minmax(std::vector<storm::ClnRationalNumber> const& values) {
            assert(!values.empty());
            storm::ClnRationalNumber min = values.front();
            storm::ClnRationalNumber max = values.front();
            for (auto const& vt : values) {
                if (vt == storm::utility::infinity<storm::ClnRationalNumber>()) {
                    max = vt;
                } else {
                    if (vt < min) {
                        min = vt;
                    }
                    if (vt > max) {
                        max = vt;
                    }
                }
            }
            return std::make_pair(min, max);
        }

        template<>
        uint_fast64_t convertNumber(ClnRationalNumber const& number){
            return carl::toInt<unsigned long>(number);
        }
        
        template<>
        ClnRationalNumber convertNumber(ClnRationalNumber const& number){
            return number;
        }
        
        template<>
        ClnRationalNumber convertNumber(double const& number){
            return carl::rationalize<ClnRationalNumber>(number);
        }

        template<>
        ClnRationalNumber convertNumber(int const& number){
            return carl::rationalize<ClnRationalNumber>(number);
        }

        template<>
        ClnRationalNumber convertNumber(uint_fast64_t const& number){
            STORM_LOG_ASSERT(static_cast<carl::uint>(number) == number, "Rationalizing failed, because the number is too large.");
            return carl::rationalize<ClnRationalNumber>(static_cast<carl::uint>(number));
        }
        
        template<>
        ClnRationalNumber convertNumber(int_fast64_t const& number){
            STORM_LOG_ASSERT(static_cast<carl::sint>(number) == number, "Rationalizing failed, because the number is too large.");
            return carl::rationalize<ClnRationalNumber>(static_cast<carl::sint>(number));
        }
        
        template<>
        double convertNumber(ClnRationalNumber const& number){
            return carl::toDouble(number);
        }
        
        template<>
        ClnRationalNumber convertNumber(std::string const& number) {
            return carl::parse<ClnRationalNumber>(number);
        }

        template<>
        ClnRationalNumber sqrt(ClnRationalNumber const& number) {
            return carl::sqrt(number);
        }
        
        template<>
        ClnRationalNumber abs(storm::ClnRationalNumber const& number) {
            return carl::abs(number);
        }
        
        template<>
        ClnRationalNumber floor(storm::ClnRationalNumber const& number) {
            return carl::floor(number);
        }
        
        template<>
        ClnRationalNumber ceil(storm::ClnRationalNumber const& number) {
            return carl::ceil(number);
        }
        
        template<>
        ClnRationalNumber pow(ClnRationalNumber const& value, uint_fast64_t exponent) {
            return carl::pow(value, exponent);
        }
#endif
        
#if defined(STORM_HAVE_CARL) && defined(STORM_HAVE_GMP)
        template<>
        storm::GmpRationalNumber infinity() {
            // FIXME: this should be treated more properly.
            return storm::GmpRationalNumber(-1);
        }
        
        template<>
        bool isOne(storm::GmpRationalNumber const& a) {
            return carl::isOne(a);
        }
        
        template<>
        bool isZero(storm::GmpRationalNumber const& a) {
            return carl::isZero(a);
        }
        
        template<>
        bool isInteger(storm::GmpRationalNumber const& number) {
            return carl::isInteger(number);
        }
        
        template<>
        std::pair<storm::GmpRationalNumber, storm::GmpRationalNumber> minmax(std::vector<storm::GmpRationalNumber> const& values) {
            assert(!values.empty());
            storm::GmpRationalNumber min = values.front();
            storm::GmpRationalNumber max = values.front();
            for (auto const& vt : values) {
                if (vt == storm::utility::infinity<storm::GmpRationalNumber>()) {
                    max = vt;
                } else {
                    if (vt < min) {
                        min = vt;
                    }
                    if (vt > max) {
                        max = vt;
                    }
                }
            }
            return std::make_pair(min, max);
        }
        
        template<>
        std::pair<storm::GmpRationalNumber, storm::GmpRationalNumber> minmax(std::map<uint64_t, storm::GmpRationalNumber> const& values) {
            assert(!values.empty());
            storm::GmpRationalNumber min = values.begin()->second;
            storm::GmpRationalNumber max = values.begin()->second;
            for (auto const& vt : values) {
                if (vt.second == storm::utility::infinity<storm::GmpRationalNumber>()) {
                    max = vt.second;
                } else {
                    if (vt.second < min) {
                        min = vt.second;
                    }
                    if (vt.second > max) {
                        max = vt.second;
                    }
                }
            }
            return std::make_pair(min, max);
        }
        
        template<>
        uint_fast64_t convertNumber(GmpRationalNumber const& number){
            return carl::toInt<unsigned long>(number);
        }
        
        template<>
        GmpRationalNumber convertNumber(GmpRationalNumber const& number){
            return number;
        }
        
        template<>
        GmpRationalNumber convertNumber(double const& number){
            return carl::rationalize<GmpRationalNumber>(number);
        }

        template<>
        GmpRationalNumber convertNumber(int const& number){
            return carl::rationalize<GmpRationalNumber>(number);
        }

        template<>
        GmpRationalNumber convertNumber(uint_fast64_t const& number){
            STORM_LOG_ASSERT(static_cast<carl::uint>(number) == number, "Rationalizing failed, because the number is too large.");
            return carl::rationalize<GmpRationalNumber>(static_cast<carl::uint>(number));
        }
        
        template<>
        GmpRationalNumber convertNumber(int_fast64_t const& number){
            STORM_LOG_ASSERT(static_cast<carl::sint>(number) == number, "Rationalizing failed, because the number is too large.");
            return carl::rationalize<GmpRationalNumber>(static_cast<carl::sint>(number));
        }
        
        template<>
        double convertNumber(GmpRationalNumber const& number){
            return carl::toDouble(number);
        }
        
        template<>
        GmpRationalNumber convertNumber(std::string const& number) {
            return carl::parse<GmpRationalNumber>(number);
        }
        
        template<>
        GmpRationalNumber sqrt(GmpRationalNumber const& number) {
            return carl::sqrt(number);
        }
        
        template<>
        GmpRationalNumber abs(storm::GmpRationalNumber const& number) {
            return carl::abs(number);
        }
        
        template<>
        GmpRationalNumber floor(storm::GmpRationalNumber const& number) {
            return carl::floor(number);
        }
        
        template<>
        GmpRationalNumber ceil(storm::GmpRationalNumber const& number) {
            return carl::ceil(number);
        }
        
        template<>
        GmpRationalNumber pow(GmpRationalNumber const& value, uint_fast64_t exponent) {
            return carl::pow(value, exponent);
        }
#endif
        
#if defined(STORM_HAVE_CARL) && defined(STORM_HAVE_GMP) && defined(STORM_HAVE_CLN)
        template<>
        storm::GmpRationalNumber convertNumber(storm::ClnRationalNumber const& number) {
            return carl::parse<storm::GmpRationalNumber>(to_string(number));
        }
        
        template<>
        storm::ClnRationalNumber convertNumber(storm::GmpRationalNumber const& number) {
            return carl::parse<storm::ClnRationalNumber>(to_string(number));
        }
#endif
        
#ifdef STORM_HAVE_CARL
        template<>
        storm::RationalFunction infinity() {
            // FIXME: this should be treated more properly.
            return storm::RationalFunction(-1.0);
        }
        
        template<>
        bool isOne(storm::RationalFunction const& a) {
            return a.isOne();
        }

        template<>
        bool isOne(storm::Polynomial const& a) {
            return a.isOne();
        }

        template<>
        bool isZero(storm::RationalFunction const& a) {
            return a.isZero();
        }

        template<>
        bool isZero(storm::Polynomial const& a) {
            return a.isZero();
        }

        template<>
        bool isConstant(storm::RationalFunction const& a) {
            return a.isConstant();
        }

        template<>
        bool isConstant(storm::Polynomial const& a) {
            return a.isConstant();
        }

        template<>
        bool isInfinity(storm::RationalFunction const& a) {
            // FIXME: this should be treated more properly.
            return a == infinity<storm::RationalFunction>();
        }

        template<>
        bool isInteger(storm::RationalFunction const& func) {
            return storm::utility::isConstant(func) && storm::utility::isOne(func.denominator());
        }
        
        template<>
        RationalFunction convertNumber(double const& number){
            return RationalFunction(carl::rationalize<RationalFunctionCoefficient>(number));
        }
        
        template<>
        RationalFunction convertNumber(int_fast64_t const& number){
            STORM_LOG_ASSERT(static_cast<carl::sint>(number) == number, "Rationalizing failed, because the number is too large.");
            return RationalFunction(carl::rationalize<RationalFunctionCoefficient>(static_cast<carl::sint>(number)));
        }
        
#if defined(STORM_HAVE_CLN)
        template<>
        RationalFunction convertNumber(ClnRationalNumber const& number) {
            return RationalFunction(convertNumber<storm::RationalFunctionCoefficient>(number));
        }
#endif
        
#if defined(STORM_HAVE_GMP)
        template<>
        RationalFunction convertNumber(GmpRationalNumber const& number) {
            return RationalFunction(convertNumber<storm::RationalFunctionCoefficient>(number));
        }
#endif
        
        template<>
        uint_fast64_t convertNumber(RationalFunction const& func) {
            return carl::toInt<unsigned long>(func.nominatorAsNumber());
        }
        
        template<>
        double convertNumber(RationalFunction const& func) {
            return carl::toDouble(func.nominatorAsNumber()) / carl::toDouble(func.denominatorAsNumber());
        }
        
        template<>
        RationalFunction convertNumber(RationalFunction const& number){
            return number;
        }

        template<>
        RationalFunctionCoefficient convertNumber(RationalFunction const& number){
            return number.nominatorAsNumber() / number.denominatorAsNumber();
        }
        
        template<>
        RationalFunction& simplify(RationalFunction& value);
        
        template<>
        RationalFunction&& simplify(RationalFunction&& value);
        
        template<>
        RationalFunction simplify(RationalFunction value) {
            value.simplify();
            return value;
        }
        
        template<>
        RationalFunction& simplify(RationalFunction& value) {
            value.simplify();
            return value;
        }
        
        template<>
        RationalFunction&& simplify(RationalFunction&& value) {
            value.simplify();
            return std::move(value);
        }
        
        template<>
        std::pair<storm::RationalFunction, storm::RationalFunction> minmax(std::vector<storm::RationalFunction> const&) {
            STORM_LOG_THROW(false, storm::exceptions::InvalidArgumentException, "Minimum/maximum for rational functions is not defined.");
        }

        template<>
        storm::RationalFunction minimum(std::vector<storm::RationalFunction> const&) {
            STORM_LOG_THROW(false, storm::exceptions::InvalidArgumentException, "Minimum for rational functions is not defined.");
        }
        
        template<>
        storm::RationalFunction maximum(std::vector<storm::RationalFunction> const&) {
            STORM_LOG_THROW(false, storm::exceptions::InvalidArgumentException, "Maximum for rational functions is not defined.");
        }
        
        template<>
        std::pair<storm::RationalFunction, storm::RationalFunction> minmax(std::map<uint64_t, storm::RationalFunction> const&) {
            STORM_LOG_THROW(false, storm::exceptions::InvalidArgumentException, "Maximum/maximum for rational functions is not defined.");
        }
        
        template<>
        storm::RationalFunction minimum(std::map<uint64_t, storm::RationalFunction> const&) {
            STORM_LOG_THROW(false, storm::exceptions::InvalidArgumentException, "Minimum for rational functions is not defined.");
        }
        
        template<>
        storm::RationalFunction maximum(std::map<uint64_t, storm::RationalFunction> const&) {
            STORM_LOG_THROW(false, storm::exceptions::InvalidArgumentException, "Maximum for rational functions is not defined");
        }

        template<>
        RationalFunction pow(RationalFunction const& value, uint_fast64_t exponent) {
            return carl::pow(value, exponent);
        }

        template<>
        std::string to_string(RationalFunction const& f) {
            std::stringstream ss;
            if (f.isConstant())  {
                if (f.denominator().isOne()) {
                    ss << f.nominatorAsNumber();
                } else {
                    ss << f.nominatorAsNumber() << "/" << f.denominatorAsNumber();
                }
            } else if (f.denominator().isOne()) {
                ss << f.nominatorAsPolynomial().coefficient() * f.nominatorAsPolynomial().polynomial();
            } else {
                ss << "(" << f.nominatorAsPolynomial() << ")/(" << f.denominatorAsPolynomial() << ")";
            }
            return ss.str();
        }

#endif
        
        
        // Explicit instantiations.
        
        // double
        template double one();
        template double zero();
        template double infinity();
        template bool isOne(double const& value);
        template bool isZero(double const& value);
        template bool isConstant(double const& value);
        template bool isInfinity(double const& value);
        template bool isInteger(double const& number);
        template double simplify(double value);
        template storm::storage::MatrixEntry<storm::storage::sparse::state_type, double> simplify(storm::storage::MatrixEntry<storm::storage::sparse::state_type, double> matrixEntry);
        template storm::storage::MatrixEntry<storm::storage::sparse::state_type, double>& simplify(storm::storage::MatrixEntry<storm::storage::sparse::state_type, double>& matrixEntry);
        template storm::storage::MatrixEntry<storm::storage::sparse::state_type, double>&& simplify(storm::storage::MatrixEntry<storm::storage::sparse::state_type, double>&& matrixEntry);
        template std::pair<double, double> minmax(std::vector<double> const&);
        template double minimum(std::vector<double> const&);
        template double maximum(std::vector<double> const&);
        template std::pair<double, double> minmax(std::map<uint64_t, double> const&);
        template double minimum(std::map<uint64_t, double> const&);
        template double maximum(std::map<uint64_t, double> const&);
        template double pow(double const& value, uint_fast64_t exponent);
        template double sqrt(double const& number);
        template double abs(double const& number);
        template double floor(double const& number);
        template double ceil(double const& number);
        template std::string to_string(double const& value);

        // float
        template float one();
        template float zero();
        template float infinity();
        template bool isOne(float const& value);
        template bool isZero(float const& value);
        template bool isConstant(float const& value);
        template bool isInfinity(float const& value);
        template bool isInteger(float const& number);
        template float simplify(float value);
        template storm::storage::MatrixEntry<storm::storage::sparse::state_type, float> simplify(storm::storage::MatrixEntry<storm::storage::sparse::state_type, float> matrixEntry);
        template storm::storage::MatrixEntry<storm::storage::sparse::state_type, float>& simplify(storm::storage::MatrixEntry<storm::storage::sparse::state_type, float>& matrixEntry);
        template storm::storage::MatrixEntry<storm::storage::sparse::state_type, float>&& simplify(storm::storage::MatrixEntry<storm::storage::sparse::state_type, float>&& matrixEntry);
        template std::pair<float, float> minmax(std::vector<float> const&);
        template float minimum(std::vector<float> const&);
        template float maximum(std::vector<float> const&);
        template std::pair<float, float> minmax(std::map<uint64_t, float> const&);
        template float minimum(std::map<uint64_t, float> const&);
        template float maximum(std::map<uint64_t, float> const&);
        template float pow(float const& value, uint_fast64_t exponent);
        template float sqrt(float const& number);
        template float abs(float const& number);
        template float floor(float const& number);
        template float ceil(float const& number);
        template std::string to_string(float const& value);

        // int
        template int one();
        template int zero();
        template int infinity();
        template bool isOne(int const& value);
        template bool isZero(int const& value);
        template bool isConstant(int const& value);
        template bool isInfinity(int const& value);
        template bool isInteger(int const& number);
        
        // uint32_t
        template uint32_t one();
        template uint32_t zero();
        template uint32_t infinity();
        template bool isOne(uint32_t const& value);
        template bool isZero(uint32_t const& value);
        template bool isConstant(uint32_t const& value);
        template bool isInfinity(uint32_t const& value);
        template bool isInteger(uint32_t const& number);
        
        // storm::storage::sparse::state_type
        template storm::storage::sparse::state_type one();
        template storm::storage::sparse::state_type zero();
        template storm::storage::sparse::state_type infinity();
        template bool isOne(storm::storage::sparse::state_type const& value);
        template bool isZero(storm::storage::sparse::state_type const& value);
        template bool isConstant(storm::storage::sparse::state_type const& value);
        template bool isInfinity(storm::storage::sparse::state_type const& value);
        template bool isInteger(storm::storage::sparse::state_type const& number);
        
#if defined(STORM_HAVE_CLN)
        // Instantiations for (CLN) rational number.
        template storm::ClnRationalNumber one();
        template storm::ClnRationalNumber zero();
        template storm::ClnRationalNumber infinity();
        template bool isOne(storm::ClnRationalNumber const& value);
        template bool isZero(storm::ClnRationalNumber const& value);
        template bool isConstant(storm::ClnRationalNumber const& value);
        template bool isInfinity(storm::ClnRationalNumber const& value);
        template bool isInteger(storm::ClnRationalNumber const& number);
        template double convertNumber(storm::ClnRationalNumber const& number);
        template uint_fast64_t convertNumber(storm::ClnRationalNumber const& number);
        template storm::ClnRationalNumber convertNumber(double const& number);
        template storm::ClnRationalNumber convertNumber(storm::ClnRationalNumber const& number);
        template ClnRationalNumber convertNumber(std::string const& number);
        template storm::ClnRationalNumber simplify(storm::ClnRationalNumber value);
        template storm::storage::MatrixEntry<storm::storage::sparse::state_type, storm::ClnRationalNumber> simplify(storm::storage::MatrixEntry<storm::storage::sparse::state_type, storm::ClnRationalNumber> matrixEntry);
        template storm::storage::MatrixEntry<storm::storage::sparse::state_type, storm::ClnRationalNumber>& simplify(storm::storage::MatrixEntry<storm::storage::sparse::state_type, storm::ClnRationalNumber>& matrixEntry);
        template storm::storage::MatrixEntry<storm::storage::sparse::state_type, storm::ClnRationalNumber>&& simplify(storm::storage::MatrixEntry<storm::storage::sparse::state_type, storm::ClnRationalNumber>&& matrixEntry);
        template std::pair<storm::ClnRationalNumber, storm::ClnRationalNumber> minmax(std::map<uint64_t, storm::ClnRationalNumber> const&);
        template storm::ClnRationalNumber minimum(std::map<uint64_t, storm::ClnRationalNumber> const&);
        template storm::ClnRationalNumber maximum(std::map<uint64_t, storm::ClnRationalNumber> const&);
        template std::pair<storm::ClnRationalNumber, storm::ClnRationalNumber> minmax(std::vector<storm::ClnRationalNumber> const&);
        template storm::ClnRationalNumber minimum(std::vector<storm::ClnRationalNumber> const&);
        template storm::ClnRationalNumber maximum(std::vector<storm::ClnRationalNumber> const&);
        template storm::ClnRationalNumber pow(storm::ClnRationalNumber const& value, uint_fast64_t exponent);
        template storm::ClnRationalNumber sqrt(storm::ClnRationalNumber const& number);
        template storm::ClnRationalNumber abs(storm::ClnRationalNumber const& number);
        template storm::ClnRationalNumber floor(storm::ClnRationalNumber const& number);
        template storm::ClnRationalNumber ceil(storm::ClnRationalNumber const& number);
        template std::string to_string(storm::ClnRationalNumber const& value);
#endif
        
#if defined(STORM_HAVE_GMP)
        // Instantiations for (GMP) rational number.
        template storm::GmpRationalNumber one();
        template storm::GmpRationalNumber zero();
        template storm::GmpRationalNumber infinity();
        template bool isOne(storm::GmpRationalNumber const& value);
        template bool isZero(storm::GmpRationalNumber const& value);
        template bool isConstant(storm::GmpRationalNumber const& value);
        template bool isInfinity(storm::GmpRationalNumber const& value);
        template bool isInteger(storm::GmpRationalNumber const& number);
        template double convertNumber(storm::GmpRationalNumber const& number);
        template uint_fast64_t convertNumber(storm::GmpRationalNumber const& number);
        template storm::GmpRationalNumber convertNumber(double const& number);
        template storm::GmpRationalNumber convertNumber(storm::GmpRationalNumber const& number);
        template GmpRationalNumber convertNumber(std::string const& number);
        template storm::GmpRationalNumber simplify(storm::GmpRationalNumber value);
        template storm::storage::MatrixEntry<storm::storage::sparse::state_type, storm::GmpRationalNumber> simplify(storm::storage::MatrixEntry<storm::storage::sparse::state_type, storm::GmpRationalNumber> matrixEntry);
        template storm::storage::MatrixEntry<storm::storage::sparse::state_type, storm::GmpRationalNumber>& simplify(storm::storage::MatrixEntry<storm::storage::sparse::state_type, storm::GmpRationalNumber>& matrixEntry);
        template storm::storage::MatrixEntry<storm::storage::sparse::state_type, storm::GmpRationalNumber>&& simplify(storm::storage::MatrixEntry<storm::storage::sparse::state_type, storm::GmpRationalNumber>&& matrixEntry);
        template std::pair<storm::GmpRationalNumber, storm::GmpRationalNumber> minmax(std::map<uint64_t, storm::GmpRationalNumber> const&);
        template storm::GmpRationalNumber minimum(std::map<uint64_t, storm::GmpRationalNumber> const&);
        template storm::GmpRationalNumber maximum(std::map<uint64_t, storm::GmpRationalNumber> const&);
        template std::pair<storm::GmpRationalNumber, storm::GmpRationalNumber> minmax(std::vector<storm::GmpRationalNumber> const&);
        template storm::GmpRationalNumber minimum(std::vector<storm::GmpRationalNumber> const&);
        template storm::GmpRationalNumber maximum(std::vector<storm::GmpRationalNumber> const&);
        template storm::GmpRationalNumber pow(storm::GmpRationalNumber const& value, uint_fast64_t exponent);
        template storm::GmpRationalNumber sqrt(storm::GmpRationalNumber const& number);
        template storm::GmpRationalNumber abs(storm::GmpRationalNumber const& number);
        template storm::GmpRationalNumber floor(storm::GmpRationalNumber const& number);
        template storm::GmpRationalNumber ceil(storm::GmpRationalNumber const& number);
        template std::string to_string(storm::GmpRationalNumber const& value);
#endif
        
#if defined(STORM_HAVE_CARL) && defined(STORM_HAVE_GMP) && defined(STORM_HAVE_CLN)
        template storm::GmpRationalNumber convertNumber(storm::ClnRationalNumber const& number);
        template storm::ClnRationalNumber convertNumber(storm::GmpRationalNumber const& number);
#endif
        
#ifdef STORM_HAVE_CARL
        // Instantiations for rational function.
        template RationalFunction one();
        template RationalFunction zero();
        template storm::RationalFunction infinity();
        template bool isOne(RationalFunction const& value);
        template bool isZero(RationalFunction const& value);
        template bool isConstant(RationalFunction const& value);
        template bool isInfinity(RationalFunction const& value);
        template bool isInteger(RationalFunction const&);
        template storm::RationalFunction convertNumber(storm::RationalFunction const& number);
        template RationalFunctionCoefficient convertNumber(RationalFunction const& number);
        template RationalFunction convertNumber(storm::RationalFunctionCoefficient const& number);
        template RationalFunction simplify(RationalFunction value);
        template RationalFunction& simplify(RationalFunction& value);
        template RationalFunction&& simplify(RationalFunction&& value);
        template storm::storage::MatrixEntry<storm::storage::sparse::state_type, RationalFunction> simplify(storm::storage::MatrixEntry<storm::storage::sparse::state_type, RationalFunction> matrixEntry);
        template storm::storage::MatrixEntry<storm::storage::sparse::state_type, RationalFunction>& simplify(storm::storage::MatrixEntry<storm::storage::sparse::state_type, RationalFunction>& matrixEntry);
        template storm::storage::MatrixEntry<storm::storage::sparse::state_type, RationalFunction>&& simplify(storm::storage::MatrixEntry<storm::storage::sparse::state_type, RationalFunction>&& matrixEntry);
        template std::pair<storm::RationalFunction, storm::RationalFunction> minmax(std::map<uint64_t, storm::RationalFunction> const&);
        template storm::RationalFunction minimum(std::map<uint64_t, storm::RationalFunction> const&);
        template storm::RationalFunction maximum(std::map<uint64_t, storm::RationalFunction> const&);
        template std::pair<storm::RationalFunction, storm::RationalFunction> minmax(std::vector<storm::RationalFunction> const&);
        template storm::RationalFunction minimum(std::vector<storm::RationalFunction> const&);
        template storm::RationalFunction maximum(std::vector<storm::RationalFunction> const&);
        template RationalFunction pow(RationalFunction const& value, uint_fast64_t exponent);

        // Instantiations for polynomials.
        template Polynomial one();
        template Polynomial zero();

        // Instantiations for intervals.
        template Interval one();
        template Interval zero();
        template bool isOne(Interval const& value);
        template bool isZero(Interval const& value);
        template bool isConstant(Interval const& value);
        template bool isInfinity(Interval const& value);
#endif
        
    }
}
