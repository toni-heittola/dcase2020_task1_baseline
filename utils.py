#!/usr/bin/env python
# -*- coding: utf-8 -*-
import dcase_util
import sys
import os
import argparse
import textwrap


def handle_application_arguments(app_parameters, raw_parameters, application_title='', version=''):
    """Handle application arguments

    Parameters
    ----------
    app_parameters : dcase_util.containers.DCASEAppParameterContainer
        Application parameters

    raw_parameters : dict
        Application parameters in dict format

    application_title : str
        Application title
        Default value ''

    version : str
        Application version
        Default value ''

    Returns
    -------
    nothing


    """

    parser = argparse.ArgumentParser(
        prefix_chars='-+',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent(
            '''\
            DCASE 2020 
            {app_title}
            Baseline system
            ---------------------------------------------            
            Author:  Toni Heittola ( toni.heittola@tuni.fi )
            Tampere University / Audio Research Group
            '''.format(app_title=application_title)
        )
    )

    # Setup argument handling
    parser.add_argument(
        '-m', '--mode',
        choices=('dev', 'eval'),
        default=None,
        help="Selector for application operation mode",
        required=False,
        dest='mode',
        type=str
    )

    # Application parameter modification
    parser.add_argument(
        '-s', '--parameter_set',
        help='Parameter set id, can be comma separated list',
        dest='parameter_set',
        required=False,
        type=str
    )

    parser.add_argument(
        '-p', '--param_file',
        help='Parameter file override',
        dest='parameter_override',
        required=False,
        metavar='FILE',
        type=dcase_util.utils.argument_file_exists
    )

    # Specific actions
    parser.add_argument(
        '--overwrite',
        help='Overwrite mode',
        dest='overwrite',
        action='store_true',
        required=False
    )

    parser.add_argument(
        '--download_dataset',
        help='Download dataset to given path and exit',
        dest='dataset_path',
        required=False,
        type=str
    )

    # Output
    parser.add_argument(
        '-o', '--output',
        help='Output file',
        dest='output_file',
        required=False,
        type=str
    )

    # Show information
    parser.add_argument(
        '--show_parameters',
        help='Show active application parameter set',
        dest='show_parameters',
        action='store_true',
        required=False
    )

    parser.add_argument(
        '--show_sets',
        help='List of available parameter sets',
        dest='show_set_list',
        action='store_true',
        required=False
    )

    parser.add_argument(
        '--show_results',
        help='Show results of the evaluated system setups',
        dest='show_results',
        action='store_true',
        required=False
    )

    # Application information
    parser.add_argument(
        '-v', '--version',
        help='Show version number and exit',
        action='version',
        version='%(prog)s ' + version
    )

    # Parse arguments
    args = parser.parse_args()

    if args.parameter_override:
        # Override parameters from a file
        app_parameters.override(override=args.parameter_override)

    overwrite = None
    if args.overwrite:
        overwrite = True

    if args.show_parameters:
        # Process parameters, and clean up parameters a bit for showing

        if args.parameter_set:
            # Check parameter set ids given as program arguments
            parameters_sets = args.parameter_set.split(',')

            for parameter_set in parameters_sets:
                # Set parameter set
                param_current = dcase_util.containers.DCASEAppParameterContainer(
                    raw_parameters,
                    path_structure={
                        'FEATURE_EXTRACTOR': ['FEATURE_EXTRACTOR'],
                        'FEATURE_NORMALIZER': ['FEATURE_EXTRACTOR'],
                        'LEARNER': ['DATA_PROCESSING_CHAIN', 'LEARNER'],
                        'RECOGNIZER': ['DATA_PROCESSING_CHAIN', 'LEARNER', 'RECOGNIZER'],
                    }
                )
                if args.parameter_override:
                    # Override parameters from a file
                    param_current.override(override=args.parameter_override)

                param_current.process(
                    create_paths=False,
                    create_parameter_hints=False
                )

                param_current['active_set'] = parameter_set
                param_current.update_parameter_set(parameter_set)
                del param_current['sets']
                del param_current['defaults']
                for section in param_current:
                    if section.endswith('_method_parameters'):
                        param_current[section] = {}

                param_current.log()
        else:
            app_parameters.process(
                create_paths=False,
                create_parameter_hints=False
            )
            del app_parameters['sets']
            del app_parameters['defaults']
            for section in app_parameters:
                if section.endswith('_method_parameters'):
                    app_parameters[section] = {}

            app_parameters.log()
        sys.exit(0)

    return args, overwrite


def save_system_output(db, folds, param, log, output_file, mode='dcase'):
    """Save system output

    Parameters
    ----------
    db : dcase_util.dataset.Dataset
        Dataset

    folds : list
        List of active folds

    param : dcase_util.containers.DCASEAppParameterContainer
        Application parameters

    log : dcase_util.ui.FancyLogger
        Logging interface

    output_file : str

    mode : str
        Output mode, possible values ['dcase', 'leaderboard']
        Default value 'dcase'

    Returns
    -------
    nothing

    """

    # Initialize results container
    all_res = dcase_util.containers.MetaDataContainer(
        filename=output_file
    )

    # Loop over all cross-validation folds and collect results
    for fold in folds:
        # Get results filename
        fold_results_filename = os.path.join(
            param.get_path('path.application.recognizer'),
            'res_fold_{fold}.csv'.format(fold=fold)
        )

        if os.path.isfile(fold_results_filename):
            # Load results container
            res = dcase_util.containers.MetaDataContainer().load(
                filename=fold_results_filename
            )
            all_res += res

        else:
            raise ValueError(
                'Results output file does not exists [{fold_results_filename}]'.format(
                    fold_results_filename=fold_results_filename
                )
            )

    if len(all_res) == 0:
        raise ValueError(
            'There are no results to output into [{output_file}]'.format(
                output_file=output_file
            )
        )

    # Convert paths to relative to the dataset root
    for item in all_res:
        item.filename = db.absolute_to_relative_path(item.filename)

        if mode == 'leaderboard':
            item['Id'] = os.path.splitext(os.path.split(item.filename)[-1])[0]
            item['Scene_label'] = item.scene_label

    if mode == 'leaderboard':
        all_res.save(fields=['Id', 'Scene_label'], delimiter=',')

    else:
        fields = ['filename', 'scene_label']
        fields += db.scene_labels()
        all_res.save(fields=fields, csv_header=True)

    log.line('System output saved to [{output_file}]'.format(output_file=output_file), indent=2)
    log.line()


def show_general_information(parameter_set, active_folds, param, db, log):
    """Show application general information

    Parameters
    ----------
    parameter_set : str
        Dataset

    active_folds : list
        List of active folds

    param : dcase_util.containers.DCASEAppParameterContainer
        Application parameters

    db : dcase_util.dataset.Dataset
        Dataset

    log : dcase_util.ui.FancyLogger
        Logging interface

    Returns
    -------
    nothing

    """

    log.section_header('General information')
    log.line('Parameter set', indent=2)
    log.data(field='Set ID', value=parameter_set, indent=4)
    log.data(field='Set description', value=param.get('description'), indent=4)

    log.line('Application', indent=2)
    log.data(field='Overwrite', value=param.get_path('general.overwrite'), indent=4)

    log.data(field='Dataset', value=db.storage_name, indent=4)
    log.data(field='Active folds', value=active_folds, indent=4)
    log.line()
    log.foot()


def show_results(param, log):
    """Show system evaluation results

    Parameters
    ----------
    param : dcase_util.containers.DCASEAppParameterContainer
        Application parameters

    log : dcase_util.ui.FancyLogger
        Logging interface

    Returns
    -------
    nothing

    """

    eval_path = param.get_path('path.application.evaluator')

    eval_files = dcase_util.utils.Path().file_list(path=eval_path, extensions='yaml')

    eval_data = {}
    for filename in eval_files:
        data = dcase_util.containers.DictContainer().load(filename=filename)
        set_id = data.get_path('parameters.set_id')
        if set_id not in eval_data:
            eval_data[set_id] = {}

        params_hash = data.get_path('parameters._hash')

        if params_hash not in eval_data[set_id]:
            eval_data[set_id][params_hash] = data

    log.section_header('Evaluated systems')
    log.line()
    log.row_reset()
    log.row(
        'Set ID', 'Mode', 'Accuracy', 'Description', 'Parameter hash',
        widths=[25, 10, 11, 45, 35],
        separators=[False, True, True, True, True],
        types=['str25', 'str10', 'float1_percentage', 'str', 'str']
    )
    log.row_sep()
    for set_id in sorted(list(eval_data.keys())):
        for params_hash in eval_data[set_id]:
            data = eval_data[set_id][params_hash]
            desc = data.get_path('parameters.description')
            application_mode = data.get_path('application_mode', '')
            log.row(
                set_id,
                application_mode,
                data.get_path('overall_accuracy') * 100.0,
                desc,
                params_hash
            )
    log.line()
    sys.exit(0)


def show_parameter_sets(param, log):
    """Show available parameter sets

    Parameters
    ----------
    param : dcase_util.containers.DCASEAppParameterContainer
        Application parameters

    log : dcase_util.ui.FancyLogger
        Logging interface

    Returns
    -------
    nothing

    """

    log.section_header('Parameter sets')
    log.line()
    log.row_reset()
    log.row(
        'Set ID', 'Description',
        widths=[50, 70],
        separators=[True, False],
    )
    log.row_sep()
    for set_id in param.set_ids():
        current_parameter_set = param.get_set(set_id=set_id)

        if current_parameter_set:
            desc = current_parameter_set.get('description', '')
        else:
            desc = ''

        log.row(
            set_id,
            desc
        )

    log.line()
