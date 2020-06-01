#!/usr/bin/env python
# -*- coding: utf-8 -*-
# DCASE 2020
# Task 1B: Low-Complexity Acoustic Scene Classification
# Baseline system
# ---------------------------------------------
# Author: Toni Heittola ( toni.heittola@tuni.fi ), Tampere University / Audio Research Group
# License: MIT

import dcase_util
import sys
import numpy
import os
import sed_eval
from task1a import do_feature_extraction, do_feature_normalization, do_learning, do_testing
from model_size_calculation import get_keras_model_size
from utils import *


__version_info__ = ('1', '0', '0')
__version__ = '.'.join(__version_info__)


def main(argv):
    # Read application default parameter file
    parameters = dcase_util.containers.DictContainer().load(
        filename='task1b.yaml'
    )

    # Initialize application parameters
    param = dcase_util.containers.DCASEAppParameterContainer(
        parameters,
        path_structure={
            'FEATURE_EXTRACTOR': ['FEATURE_EXTRACTOR'],
            'FEATURE_NORMALIZER': ['FEATURE_EXTRACTOR'],
            'LEARNER': ['DATA_PROCESSING_CHAIN', 'LEARNER'],
            'RECOGNIZER': ['DATA_PROCESSING_CHAIN', 'LEARNER', 'RECOGNIZER'],
        }
    )

    # Handle application arguments
    args, overwrite = handle_application_arguments(
        app_parameters=param,
        raw_parameters=parameters,
        application_title='Task 1B: Low-Complexity Acoustic Scene Classification',
        version=__version__
    )

    # Process parameters, this is done only after application argument handling in case
    # parameters where injected from command line.
    param.process()

    if args.parameter_set:
        # Check parameter set ids given as program arguments
        parameters_sets = args.parameter_set.split(',')

        # Check parameter_sets
        for set_id in parameters_sets:
            if not param.set_id_exists(set_id=set_id):
                raise ValueError('Parameter set id [{set_id}] not found.'.format(set_id=set_id))

    else:
        parameters_sets = [param.active_set()]

    # Get application mode
    if args.mode:
        application_mode = args.mode

    else:
        application_mode = 'dev'

    if args.dataset_path:
        # Download only dataset if requested

        # Make sure given path exists
        dcase_util.utils.Path().create(
            paths=args.dataset_path
        )

        for parameter_set in parameters_sets:
            # Set parameter set
            param['active_set'] = parameter_set
            param.update_parameter_set(parameter_set)

            if application_mode == 'eval':
                eval_parameter_set_id = param.active_set() + '_eval'
                if not param.set_id_exists(eval_parameter_set_id):
                    raise ValueError(
                        'Parameter set id [{set_id}] not found for eval mode.'.format(
                            set_id=eval_parameter_set_id
                        )
                    )

                # Change active parameter set
                param.update_parameter_set(eval_parameter_set_id)

            # Get dataset and initialize
            dcase_util.datasets.dataset_factory(
                dataset_class_name=param.get_path('dataset.parameters.dataset'),
                data_path=args.dataset_path,
            ).initialize().log()

        sys.exit(0)

    # Get overwrite flag
    if overwrite is None:
        overwrite = param.get_path('general.overwrite')

    # Make sure all system paths exists
    dcase_util.utils.Path().create(
        paths=list(param['path'].values())
    )

    # Setup logging
    dcase_util.utils.setup_logging(
        logging_file=os.path.join(param.get_path('path.log'), 'task1b.log')
    )

    # Get logging interface
    log = dcase_util.ui.ui.FancyLogger()

    # Log title
    log.title('DCASE2020 / Task1B -- Low-Complexity Acoustic Scene Classification')
    log.line()

    if args.show_results:
        # Show evaluated systems
        show_results(param=param, log=log)
        sys.exit(0)

    if args.show_set_list:
        show_parameter_sets(param=param, log=log)
        sys.exit(0)

    # Create timer instance
    timer = dcase_util.utils.Timer()

    for parameter_set in parameters_sets:
        # Set parameter set
        param['active_set'] = parameter_set
        param.update_parameter_set(parameter_set)

        # Get dataset and initialize
        db = dcase_util.datasets.dataset_factory(
            dataset_class_name=param.get_path('dataset.parameters.dataset'),
            data_path=param.get_path('path.dataset'),
        ).initialize()

        if application_mode == 'eval':
            # Application is set to work in 'eval' mode. In this mode, training is done with
            # all data from development dataset, and testing with all data from evaluation dataset.

            # Make sure we are using all data
            active_folds = db.folds(
                mode='full'
            )

        else:
            # Application working in normal mode aka 'dev' mode

            # Get active folds from dataset
            active_folds = db.folds(
                mode=param.get_path('dataset.parameters.evaluation_mode')
            )

            # Get active fold list from parameters
            active_fold_list = param.get_path('general.active_fold_list')

            if active_fold_list and len(set(active_folds).intersection(active_fold_list)) > 0:
                # Active fold list is set and it intersects with active_folds given by dataset class
                active_folds = list(set(active_folds).intersection(active_fold_list))

        # Print some general information
        show_general_information(
            parameter_set=parameter_set,
            active_folds=active_folds,
            param=param,
            db=db,
            log=log
        )

        if param.get_path('flow.feature_extraction'):
            # Feature extraction stage
            log.section_header('Feature Extraction')

            timer.start()

            processed_items = do_feature_extraction(
                db=db,
                param=param,
                log=log,
                overwrite=overwrite
            )

            timer.stop()

            log.foot(
                time=timer.elapsed(),
                item_count=len(processed_items)
            )

        if param.get_path('flow.feature_normalization'):
            # Feature extraction stage
            log.section_header('Feature Normalization')

            timer.start()

            processed_items = do_feature_normalization(
                db=db,
                folds=active_folds,
                param=param,
                log=log,
                overwrite=overwrite
            )
            timer.stop()

            log.foot(
                time=timer.elapsed(),
                item_count=len(processed_items)
            )

        if param.get_path('flow.learning'):
            # Learning stage
            log.section_header('Learning')

            timer.start()

            processed_items = do_learning(
                db=db,
                folds=active_folds,
                param=param,
                log=log,
                overwrite=overwrite
            )

            timer.stop()

            log.foot(
                time=timer.elapsed(),
                item_count=len(processed_items)
            )

        if application_mode == 'dev':
            # System evaluation in 'dev' mode

            if param.get_path('flow.testing'):
                # Testing stage
                log.section_header('Testing')

                timer.start()

                processed_items = do_testing(
                    db=db,
                    scene_labels=db.scene_labels(),
                    folds=active_folds,
                    param=param,
                    log=log,
                    overwrite=overwrite
                )

                timer.stop()

                log.foot(
                    time=timer.elapsed(),
                    item_count=len(processed_items)
                )

                if args.output_file:
                    save_system_output(
                        db=db,
                        folds=active_folds,
                        param=param,
                        log=log,
                        output_file=args.output_file
                    )

            if param.get_path('flow.evaluation'):
                # Evaluation stage
                log.section_header('Evaluation')

                timer.start()

                do_evaluation(
                    db=db,
                    folds=active_folds,
                    param=param,
                    log=log,
                    application_mode=application_mode
                )
                timer.stop()

                log.foot(
                    time=timer.elapsed(),
                )

            if param.get_path('flow.calculate_model_size'):
                log.section_header('Model size calculation')
                
                timer.start()
                
                do_model_size_calculation(
                    db=db,
                    folds=active_folds,
                    param=param,
                    log=log,
                )
                log.foot(
                    time=timer.elapsed(),
                )

        elif application_mode == 'eval':
            # System evaluation in eval mode

            # Get set id for eval parameters, test if current set id with eval post fix exists
            eval_parameter_set_id = param.active_set() + '_eval'
            if not param.set_id_exists(eval_parameter_set_id):
                raise ValueError(
                    'Parameter set id [{set_id}] not found for eval mode.'.format(
                        set_id=eval_parameter_set_id
                    )
                )

            # Change active parameter set
            param.update_parameter_set(eval_parameter_set_id)

            # Get eval dataset and initialize
            db_eval = dcase_util.datasets.dataset_factory(
                dataset_class_name=param.get_path('dataset.parameters.dataset'),
                data_path=param.get_path('path.dataset'),
            ).initialize()

            # Get active folds
            active_folds = db_eval.folds(
                mode='full'
            )

            if param.get_path('flow.feature_extraction'):
                # Feature extraction for eval
                log.section_header('Feature Extraction')

                timer.start()

                processed_items = do_feature_extraction(
                    db=db_eval,
                    param=param,
                    log=log,
                    overwrite=overwrite
                )

                timer.stop()

                log.foot(
                    time=timer.elapsed(),
                    item_count=len(processed_items)
                )

            if param.get_path('flow.testing'):
                # Testing stage for eval
                log.section_header('Testing')

                timer.start()

                processed_items = do_testing(
                    db=db_eval,
                    scene_labels=db.scene_labels(),
                    folds=active_folds,
                    param=param,
                    log=log,
                    overwrite=overwrite
                )

                timer.stop()

                log.foot(
                    time=timer.elapsed(),
                    item_count=len(processed_items)
                )

                if args.output_file:
                    save_system_output(
                        db=db_eval,
                        folds=active_folds,
                        param=param,
                        log=log,
                        output_file=args.output_file,
                        mode='dcase'
                    )

            if db_eval.reference_data_present and param.get_path('flow.evaluation'):
                if application_mode == 'eval':
                    # Evaluation stage for eval
                    log.section_header('Evaluation')

                    timer.start()

                    do_evaluation_task1b_eval(
                        db=db_eval,
                        folds=active_folds,
                        param=param,
                        log=log,
                        application_mode=application_mode
                    )

                    timer.stop()

                    log.foot(
                        time=timer.elapsed(),
                    )

    return 0

def do_evaluation(db, folds, param, log, application_mode='default'):
    """Evaluation stage

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

    application_mode : str
        Application mode
        Default value 'default'

    Returns
    -------
    nothing

    """

    all_results = []

    class_wise_results = numpy.zeros((1, len(db.scene_labels())))
    fold = 1

    fold_results_filename = os.path.join(
        param.get_path('path.application.recognizer'),
        'res_fold_{fold}.csv'.format(fold=fold)
    )

    reference_scene_list = db.eval(fold=fold)

    for item_id, item in enumerate(reference_scene_list):
        reference_scene_list[item_id]['filename'] = os.path.split(item.filename)[-1]
        reference_scene_list[item_id]['file'] = item.filename

    estimated_scene_list = dcase_util.containers.MetaDataContainer().load(
        filename=fold_results_filename,
        file_format=dcase_util.utils.FileFormat.CSV,
        csv_header=True,
        delimiter='\t'
    )

    for item_id, item in enumerate(estimated_scene_list):
        estimated_scene_list[item_id]['filename'] = os.path.split(item.filename)[-1]
        estimated_scene_list[item_id]['file'] = item.filename

    evaluator = sed_eval.scene.SceneClassificationMetrics(
        scene_labels=db.scene_labels()
    )

    evaluator.evaluate(
        reference_scene_list=reference_scene_list,
        estimated_scene_list=estimated_scene_list
    )

    # Collect data for log loss calculation
    y_true = []
    y_pred = []

    y_true_scene = {}
    y_pred_scene = {}

    estimated_scene_items = {}
    for item in estimated_scene_list:
        estimated_scene_items[item.filename] = item

    scene_labels = db.scene_labels()
    for item in reference_scene_list:
        # Find corresponding item from estimated_scene_list
        estimated_item = estimated_scene_items[item.filename]

        # Get class id
        scene_label_id = scene_labels.index(item.scene_label)
        y_true.append(scene_label_id)

        # Get class-wise probabilities in correct order
        item_probabilities = []
        for scene_label in scene_labels:
            item_probabilities.append(estimated_item[scene_label])

        y_pred.append(item_probabilities)

        if item.scene_label not in y_true_scene:
            y_true_scene[item.scene_label] = []
            y_pred_scene[item.scene_label] = []

        y_true_scene[item.scene_label].append(scene_label_id)
        y_pred_scene[item.scene_label].append(item_probabilities)

    from sklearn.metrics import log_loss
    logloss_overall = log_loss(y_true=y_true, y_pred=y_pred)

    logloss_class_wise = {}
    for scene_label in db.scene_labels():
        logloss_class_wise[scene_label] = log_loss(y_true=y_true_scene[scene_label], y_pred=y_pred_scene[scene_label], labels=list(range(len(db.scene_labels()))))

    results = evaluator.results()
    all_results.append(results)

    for scene_label_id, scene_label in enumerate(db.scene_labels()):
        class_wise_results[0, scene_label_id] = results['class_wise'][scene_label]['accuracy']['accuracy']

    overall = [
        results['class_wise_average']['accuracy']['accuracy']
    ]

    # Get filename
    filename = 'eval_{parameter_hash}_{application_mode}.yaml'.format(
        parameter_hash=param['_hash'],
        application_mode=application_mode
    )

    # Get current parameters
    current_param = dcase_util.containers.AppParameterContainer(param.get_set(param.active_set()))
    current_param._clean_unused_parameters()

    if current_param.get_path('learner.parameters.compile.optimizer'):
        current_param.set_path('learner.parameters.compile.optimizer', None)

    # Save evaluation information
    dcase_util.containers.DictContainer(
        {
            'application_mode': application_mode,
            'set_id': param.active_set(),
            'class_wise_results': class_wise_results.tolist(),
            'overall_accuracy': overall[0],
            'overall_logloss': logloss_overall,
            'all_results': all_results,
            'classwise_logloss': logloss_class_wise,
            'parameters': current_param
        }
    ).save(
        filename=os.path.join(param.get_path('path.application.evaluator'), filename)
    )

    log.line()
    log.row_reset()

    # Table header
    column_headers = ['Scene', 'Accuracy', 'Logloss']
    column_widths = [16, 13, 13]
    column_types = ['str20', 'float1_percentage', 'float3']
    column_separators = [True, True, False]

    log.row(
        *column_headers,
        widths=column_widths,
        types=column_types,
        separators=column_separators,
        indent=2
    )
    log.row_sep()

    # Class-wise rows
    for scene_label_id, scene_label in enumerate(db.scene_labels()):
        log.row(scene_label, class_wise_results[0, scene_label_id] * 100.0, logloss_class_wise[scene_label])

    log.row_sep()

    # Last row
    column_values = ['Average']
    for value in overall:
        column_values.append(value*100.0)
    column_values.append(logloss_overall)

    log.row(
        *column_values,
        types=column_types
    )

    log.line()


def do_model_size_calculation(db, folds, param, log):
    """Model size calculation stage

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

    Returns
    -------
    nothing

    """

    for fold in folds:
        log.line(
            data='Fold [{fold}]'.format(fold=fold),
            indent=2
        )

        # Get model filename
        fold_model_filename = os.path.join(
            param.get_path('path.application.learner'),
            'model_fold_{fold}.h5'.format(fold=fold)
        )

        # Setup keras
        dcase_util.keras.setup_keras(
            seed=param.get_path('learner.parameters.random_seed'),
            profile=param.get_path('learner.parameters.keras_profile'),
            backend=param.get_path('learner.parameters.backend', 'tensorflow'),
            print_indent=2
        )
        import keras            

        feature_method = param.get_path('feature_extractor.parameters.method', 'mel')
        embedding_size = None
        if feature_method == 'openl3':
            extractor = dcase_util.features.OpenL3Extractor(
                **param.get_path('feature_extractor.parameters', {})
            )
            log.section_header('Audio Embedding / OpenL3')
            embedding_size = get_keras_model_size(extractor.model, verbose=True, ui=log)

        elif feature_method == 'edgel3':
            extractor = dcase_util.features.EdgeL3Extractor(
                **param.get_path('feature_extractor.parameters', {})
            )
            log.section_header('Audio Embedding / EdgeL3')
            embedding_size = get_keras_model_size(extractor.model, verbose=True, ui=log)

        # Load acoustic model
        keras_model = keras.models.load_model(fold_model_filename)

        log.section_header('Acoustic model')
        model_size = get_keras_model_size(keras_model=keras_model, verbose=True, ui=log)

        log.row_reset()
        log.row('', 'param', 'non-zero','sparsity', 'size', widths=[25, 20, 20, 20, 20])
        log.row_sep()
        if embedding_size:
            log.row(
                'Audio embedding',
                embedding_size['parameters']['all']['count'],
                embedding_size['parameters']['non_zero']['count'],
                (1- (embedding_size['parameters']['non_zero']['count']/float(embedding_size['parameters']['all']['count']))) * 100.0,
                dcase_util.utils.get_byte_string(embedding_size['parameters']['non_zero']['bytes'], show_bytes=False)
            )

        log.row('Acoustic model',
                model_size['parameters']['all']['count'],
                model_size['parameters']['non_zero']['count'],
                (1- (model_size['parameters']['non_zero']['count']/float(model_size['parameters']['all']['count']))) * 100.0,
                dcase_util.utils.get_byte_string(model_size['parameters']['non_zero']['bytes'], show_bytes=False)
        )
        
        total_size = model_size['parameters']['non_zero']['bytes']
        if embedding_size:
            total_size += embedding_size['parameters']['non_zero']['bytes']

        log.row(
            'Total',
            '',
            '',
            '',
            dcase_util.utils.get_byte_string(total_size, show_bytes=False)
        )


if __name__ == "__main__":
    sys.exit(main(sys.argv))
