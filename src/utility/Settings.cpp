/*
 * Settings.cpp
 *
 *  Created on: 22.11.2012
 *      Author: Gereon Kremer
 */

#include "src/utility/Settings.h"

#include <boost/algorithm/string/join.hpp>
#include <utility>
#include <map>
#include <string>
#include <list>

#include "src/exceptions/BaseException.h"
#include "log4cplus/logger.h"
#include "log4cplus/loggingmacros.h"
extern log4cplus::Logger logger;


namespace storm {
namespace settings {

namespace bpo = boost::program_options;

/*
 * Static initializers.
 */
std::unique_ptr<bpo::options_description> storm::settings::Settings::desc = nullptr;
std::string storm::settings::Settings::binaryName = "";
storm::settings::Settings* storm::settings::Settings::inst = nullptr;

std::map< std::pair<std::string, std::string>, std::shared_ptr<bpo::options_description> > storm::settings::Settings::modules;

storm::settings::Destroyer storm::settings::Settings::destroyer;

/*!
 *	The constructor fills the option descriptions, parses the
 *	command line and the config file and puts the option values to
 *	our option mapping.
 *
 *	If a configfile is set in the commandline, we load this one.
 *	Otherwise, if filename is not NULL, we load this one. Otherwise,
 *	we load no config file.
 *
 *	@param argc should be argc passed to main function
 *	@param argv should be argv passed to main function
 *	@param filename	either nullptr or name of config file
 */
Settings::Settings(int const argc, char const * const argv[], char const * const filename, bool const sloppy) {
	Settings::binaryName = std::string(argv[0]);
	try {
		// Initially fill description objects.
		this->initDescriptions();

		// Check module triggers, add corresponding options.
		std::map< std::string, std::list< std::string > > options;

		for (auto it : Settings::modules) {
			options[it.first.first].push_back(it.first.second);
		}
		for (auto it : options) {
			std::stringstream str;
			str << "select " << it.first << " module (" << boost::algorithm::join(it.second, ", ") << ")";

			Settings::desc->add_options()
				(it.first.c_str(), bpo::value<std::string>()->default_value(it.second.front()), str.str().c_str())
			;
		}

		// Perform first parse run.
		this->firstRun(argc, argv, filename);

		// Buffer for items to be deleted.
		std::list< std::pair< std::string, std::string > > deleteQueue;
		// Check module triggers.
		for (auto it : Settings::modules) {
			std::pair< std::string, std::string > trigger = it.first;
			if (this->vm.count(trigger.first)) {
				if (this->vm[trigger.first].as<std::string>().compare(trigger.second) == 0) {
					Settings::desc->add(*it.second);
					deleteQueue.push_back(trigger);
				}
			}
		}
		for (auto it : deleteQueue) Settings::modules.erase(it);

		// Stop if help is set.
		if (this->vm.count("help") > 0) {
			return;
		}

		// Perform second run.
		this->secondRun(argc, argv, filename);

		// Finalize parsed options, check for specified requirements.
		if (!sloppy) {
			bpo::notify(this->vm);
		}
		LOG4CPLUS_DEBUG(logger, "Finished loading config.");
	}
	catch (bpo::reading_file const& e) {
		std::cerr << "Could not read config file " << filename << std::endl;
		LOG4CPLUS_ERROR(logger, "Could not read config file");
	}
	catch (bpo::required_option const& e) {
		throw storm::exceptions::InvalidSettingsException() << "Required option missing";
	}
	catch (bpo::validation_error const& e) {
		throw storm::exceptions::InvalidSettingsException() << "Validation failed: " << e.what();
	}
	catch (bpo::invalid_command_line_syntax const& e) {
		throw storm::exceptions::InvalidSettingsException() << e.what();
	}
	catch (bpo::error const& e) {
		throw storm::exceptions::InvalidSettingsException() << e.what();
	}
}

void checkExplicit(const std::vector<std::string>& filenames) {
	if (filenames.size() != 2) {
		throw storm::exceptions::InvalidSettingsException() << "--explicit must be given exactly two filenames";
	}
}

    /*!
     * Validates whether the given lemethod matches one of the available ones.
     * Throws an exception of type InvalidSettings in case the selected method is illegal.
     */
    static void validateLeMethod(const std::string& lemethod) {
        if ((lemethod != "bicgstab") && (lemethod != "qmr") && (lemethod != "jacobi") && (lemethod != "lscg") && (lemethod != "gmres")) {
            throw exceptions::InvalidSettingsException() << "Argument " << lemethod << " for option 'lemethod' is invalid.";
        }
    }
    
    /*!
     * Validates whether the given preconditioner matches one of the available ones.
     * Throws an exception of type InvalidSettings in case the selected preconditioner is illegal.
     */
    static void validatePreconditioner(const std::string& preconditioner) {
        if ((preconditioner != "ilu") && (preconditioner != "diagonal") && (preconditioner != "ildlt") && (preconditioner != "none")) {
            throw exceptions::InvalidSettingsException() << "Argument " << preconditioner << " for option 'precond' is invalid.";
        }
    }
    
/*!
 *	Initially fill options_description objects.
 */
void Settings::initDescriptions() {
	LOG4CPLUS_DEBUG(logger, "Initializing descriptions.");
	Settings::desc = std::unique_ptr<bpo::options_description>(new bpo::options_description("Generic Options"));
	Settings::desc->add_options()
		("help,h", "produce help message")
		("verbose,v", "be verbose")
		("debug", "be very verbose, intended for debugging")
		("trace", "be extremely verbose, expect lots of output")
		("logfile,l", bpo::value<std::string>(), "name of the log file")
		("configfile,c", bpo::value<std::string>(), "name of config file")
		("explicit", bpo::value<std::vector<std::string>>()->multitoken()->notifier(&checkExplicit), "name of transition and labeling file")
		("symbolic", bpo::value<std::string>(), "name of prism file")
		("prctl", bpo::value<std::string>(), "text file containing prctl formulas")
		("csl", bpo::value<std::string>(), "text file containing csl formulas")
		("ltl", bpo::value<std::string>(), "text file containing ltl formulas")
		("transrew", bpo::value<std::string>()->default_value(""), "name of transition reward file")
		("staterew", bpo::value<std::string>()->default_value(""), "name of state reward file")
		("fix-deadlocks", "insert self-loops for states without outgoing transitions")
        ("lemethod", boost::program_options::value<std::string>()->default_value("bicgstab")->notifier(&storm::settings::validateLeMethod), "Sets the method used for linear equation solving. Must be in {bicgstab, qmr, lscg, gmres, jacobi}.")
        ("maxiter", boost::program_options::value<unsigned>()->default_value(10000), "Sets the maximal number of iterations for iterative equation solving.")
        ("precision", boost::program_options::value<double>()->default_value(1e-6), "Sets the precision for iterative equation solving.")
        ("precond", boost::program_options::value<std::string>()->default_value("ilu")->notifier(&validatePreconditioner), "Sets the preconditioning technique for linear equation solving. Must be in {ilu, diagonal, ildlt, none}.")
        ("relative", boost::program_options::value<bool>()->default_value(true), "Sets whether the relative or absolute error is considered for detecting convergence.")
        ("use-heuristic-presolve", boost::program_options::value<bool>()->default_value(false), "Sets whether heuristic methods should be applied to get better initial values for value iteration.")
        ("matrixlib", boost::program_options::value<std::string>()->default_value("gmm++"), "Sets which matrix library is to be used for numerical solving.")
	;
}

/*!
 *	Perform a sloppy parsing run: parse command line and config file (if
 *	given), but allow for unregistered options, do not check requirements
 *	from options_description objects.
 */
void Settings::firstRun(int const argc, char const * const argv[], char const * const filename) {
	LOG4CPLUS_DEBUG(logger, "Performing first run.");
	// Parse command line.
	bpo::store(bpo::command_line_parser(argc, argv).options(*(Settings::desc)).allow_unregistered().run(), this->vm);

	/*
	 *	Load config file if specified.
	 */
	if (this->vm.count("configfile")) {
		bpo::store(bpo::parse_config_file<char>(this->vm["configfile"].as<std::string>().c_str(), *(Settings::desc)), this->vm, true);
	} else if (filename != NULL) {
		bpo::store(bpo::parse_config_file<char>(filename, *(Settings::desc)), this->vm, true);
	}
}

/*!
 *	Perform the second parser run: parse command line and config file (if
 *	given) and check for unregistered options, requirements from
 *	options_description objects.
 */
void Settings::secondRun(int const argc, char const * const argv[], char const * const filename) {
	LOG4CPLUS_DEBUG(logger, "Performing second run.");
	// Parse command line.
	bpo::store(bpo::command_line_parser(argc, argv).options(*(Settings::desc)).run(), this->vm);
	/*
	 *	Load config file if specified.
	 */
	if (this->vm.count("configfile")) {
		bpo::store(bpo::parse_config_file<char>(this->vm["configfile"].as<std::string>().c_str(), *(Settings::desc)), this->vm, true);
	} else if (filename != NULL) {
		bpo::store(bpo::parse_config_file<char>(filename, *(Settings::desc)), this->vm, true);
	}
}


/*!
 *	Print a short general usage information consisting of the the list of available command line options.
 *
 *	Use it like this:
 *	@code std::cout << storm::settings::help; @endcode
 */
std::ostream& help(std::ostream& os) {
	os << "Usage: " << storm::settings::Settings::binaryName << " [options] <transition file> <label file>" << std::endl;
	os << *(storm::settings::Settings::desc) << std::endl;
	for (auto it : Settings::modules) {
		os << *(it.second) << std::endl;
	}
	return os;
}

}  // namespace settings
}  // namespace storm
