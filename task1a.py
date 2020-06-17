#!/usr/bin/env python
# -*- coding: utf-8 -*-
# DCASE 2020
# Task 1A: Acoustic Scene Classification with Multiple Devices
# Baseline system
# ---------------------------------------------
# Author: Toni Heittola ( toni.heittola@tuni.fi ), Tampere University / Audio Research Group
# License: MIT

import dcase_util
import sys
import numpy
import os
import sed_eval
from utils import *

__version_info__ = ('1', '0', '0')
__version__ = '.'.join(__version_info__)


def main(argv):
    # Read application default parameter file
    parameters = dcase_util.containers.DictContainer().load(
        filename='task1a.yaml'
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
        application_title='Task 1A: Acoustic Scene Classification with Multiple Devices',
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
        logging_file=os.path.join(param.get_path('path.log'), 'task1a.log')
    )

    # Get logging interface
    log = dcase_util.ui.ui.FancyLogger()

    # Log title
    log.title('DCASE2020 / Task1A -- Acoustic Scene Classification with Multiple Devices')
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
            # Application is set to work in 'eval' mode. In this modes, training is done with
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

                    do_evaluation_task1a_eval(
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


def do_feature_extraction(db, param, log, overwrite=False):
    """Feature extraction stage

    Parameters
    ----------
    db : dcase_util.dataset.Dataset
        Dataset

    param : dcase_util.containers.DCASEAppParameterContainer
        Application parameters

    log : dcase_util.ui.FancyLogger
        Logging interface

    overwrite : bool
        Overwrite data always
        Default value False

    Returns
    -------
    list of str

    """

    extraction_needed = False
    processed_files = []

    if overwrite:
        extraction_needed = True
    else:
        for item_id, audio_filename in enumerate(db.audio_files):
            # Get filename for feature data from audio filename
            feature_filename = dcase_util.utils.Path(
                path=audio_filename
            ).modify(
                path_base=param.get_path('path.application.feature_extractor'),
                filename_extension='.cpickle'
            )

            if not os.path.isfile(feature_filename):
                extraction_needed = True
                break

    if extraction_needed:
        # Prepare feature extractor
        method = param.get_path('feature_extractor.parameters.method', 'mel')
        if method == 'openl3':
            extractor = dcase_util.features.OpenL3Extractor(
                **param.get_path('feature_extractor.parameters', {})
            )

        elif method == 'edgel3':
            extractor = dcase_util.features.EdgeL3Extractor(
                **param.get_path('feature_extractor.parameters', {})
            )

        elif method == 'mel':
            extractor = dcase_util.features.MelExtractor(
                **param.get_path('feature_extractor.parameters', {})
            )

        else:
            raise ValueError('Unknown feature extractor method [{method}].'.format(method=method))

        # Loop over all audio files in the current dataset and extract acoustic features for each of them.
        for item_id, audio_filename in enumerate(db.audio_files):
            # Get filename for feature data from audio filename
            feature_filename = dcase_util.utils.Path(
                path=audio_filename
            ).modify(
                path_base=param.get_path('path.application.feature_extractor'),
                filename_extension='.cpickle'
            )

            if not os.path.isfile(feature_filename) or overwrite:
                log.line(
                    data='[{item: >5} / {total}] [{filename}]'.format(
                        item=item_id,
                        total=len(db.audio_files),
                        filename=os.path.split(audio_filename)[1]
                    ),
                    indent=2
                )

                # Load audio data
                audio = dcase_util.containers.AudioContainer().load(
                    filename=audio_filename,
                    mono=True,
                    fs=param.get_path('feature_extractor.fs')
                )

                # Extract features and store them into FeatureContainer, and save it to the disk
                dcase_util.containers.FeatureContainer(
                    data=extractor.extract(audio.data),
                    time_resolution=param.get_path('feature_extractor.hop_length_seconds')
                ).save(
                    filename=feature_filename
                )

                processed_files.append(feature_filename)

    return processed_files


def do_feature_normalization(db, folds, param, log, overwrite=False):
    """Feature normalization stage

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

    overwrite : bool
        Overwrite data always
        Default value False

    Returns
    -------
    list of str

    """

    # Loop over all active cross-validation folds and calculate mean and std for the training data

    processed_files = []

    for fold in folds:
        log.line(
            data='Fold [{fold}]'.format(fold=fold),
            indent=2
        )

        # Get filename for the normalization factors
        fold_stats_filename = os.path.join(
            param.get_path('path.application.feature_normalizer'),
            'norm_fold_{fold}.cpickle'.format(fold=fold)
        )

        if not os.path.isfile(fold_stats_filename) or overwrite:
            normalizer = dcase_util.data.Normalizer(
                filename=fold_stats_filename
            )

            # Loop through all training data
            for item in db.train(fold=fold):
                # Get feature filename
                feature_filename = dcase_util.utils.Path(
                    path=item.filename
                ).modify(
                    path_base=param.get_path('path.application.feature_extractor'),
                    filename_extension='.cpickle'
                )

                # Load feature matrix
                features = dcase_util.containers.FeatureContainer().load(
                    filename=feature_filename
                )

                # Accumulate statistics
                normalizer.accumulate(
                    data=features
                )

            # Finalize and save
            normalizer.finalize().save()

            processed_files.append(fold_stats_filename)

    return processed_files


def do_learning(db, folds, param, log, overwrite=False):
    """Learning stage

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

    overwrite : bool
        Overwrite data always
        Default value False

    Returns
    -------
    nothing

    """

    # Loop over all cross-validation folds and learn acoustic models

    processed_files = []

    for fold in folds:
        log.line(
            data='Fold [{fold}]'.format(fold=fold),
            indent=2
        )


        fold_model_filename = os.path.join(
            param.get_path('path.application.learner'),
            'model_fold_{fold}.h5'.format(fold=fold)
        )

        if not os.path.isfile(fold_model_filename) or overwrite:
            log.line()

            # Create data processing chain for features
            data_processing_chain = dcase_util.processors.ProcessingChain()
            for chain in param.get_path('data_processing_chain.parameters.chain'):
                processor_name = chain.get('processor_name')
                init_parameters = chain.get('init_parameters', {})

                # Inject parameters
                if processor_name == 'dcase_util.processors.NormalizationProcessor':
                    # Get model filename
                    init_parameters['filename'] = os.path.join(
                        param.get_path('path.application.feature_normalizer'),
                        'norm_fold_{fold}.cpickle'.format(fold=fold)
                    )

                data_processing_chain.push_processor(
                    processor_name=processor_name,
                    init_parameters=init_parameters,
                )

            # Create meta processing chain for reference data
            meta_processing_chain = dcase_util.processors.ProcessingChain()
            for chain in param.get_path('meta_processing_chain.parameters.chain'):
                processor_name = chain.get('processor_name')
                init_parameters = chain.get('init_parameters', {})

                # Inject parameters
                if processor_name == 'dcase_util.processors.OneHotEncodingProcessor':
                    init_parameters['label_list'] = db.scene_labels()

                meta_processing_chain.push_processor(
                    processor_name=processor_name,
                    init_parameters=init_parameters,
                )

            if param.get_path('learner.parameters.validation_set') and param.get_path('learner.parameters.validation_set.enable', True):
                # Get validation files
                training_files, validation_files = db.validation_split(
                    fold=fold,
                    split_type='balanced',
                    validation_amount=param.get_path('learner.parameters.validation_set.validation_amount'),
                    balancing_mode=param.get_path('learner.parameters.validation_set.balancing_mode'),
                    seed=param.get_path('learner.parameters.validation_set.seed', 0),
                    verbose=True
                )

            else:
                # No validation set used
                training_files = db.train(fold=fold).unique_files
                validation_files = dcase_util.containers.MetaDataContainer()

            # Create item_list_train and item_list_validation
            item_list_train = []
            item_list_validation = []
            for item in db.train(fold=fold):
                # Get feature filename
                feature_filename = dcase_util.utils.Path(
                    path=item.filename
                ).modify(
                    path_base=param.get_path('path.application.feature_extractor'),
                    filename_extension='.cpickle'
                )

                item_ = {
                    'data': {
                        'filename': feature_filename
                    },
                    'meta': {
                        'label': item.scene_label
                    }
                }

                if item.filename in validation_files:
                    item_list_validation.append(item_)

                elif item.filename in training_files:
                    item_list_train.append(item_)

            # Setup keras, run only once
            dcase_util.keras.setup_keras(
                seed=param.get_path('learner.parameters.random_seed'),
                profile=param.get_path('learner.parameters.keras_profile'),
                backend=param.get_path('learner.parameters.backend', 'tensorflow'),
                print_indent=2
            )

            if param.get_path('learner.parameters.generator.enable'):
                # Create data generators for training and validation

                # Get generator class, class is inherited from keras.utils.Sequence class.
                KerasDataSequence = dcase_util.keras.get_keras_data_sequence_class()

                # Training data generator
                train_data_sequence = KerasDataSequence(
                    item_list=item_list_train,
                    data_processing_chain=data_processing_chain,
                    meta_processing_chain=meta_processing_chain,
                    batch_size=param.get_path('learner.parameters.fit.batch_size'),
                    data_format=param.get_path('learner.parameters.data.data_format'),
                    target_format=param.get_path('learner.parameters.data.target_format'),
                    **param.get_path('learner.parameters.generator', default={})
                )

                # Show data properties
                train_data_sequence.log()

                if item_list_validation:
                    # Validation data generator
                    validation_data_sequence = KerasDataSequence(
                        item_list=item_list_validation,
                        data_processing_chain=data_processing_chain,
                        meta_processing_chain=meta_processing_chain,
                        batch_size=param.get_path('learner.parameters.fit.batch_size'),
                        data_format=param.get_path('learner.parameters.data.data_format'),
                        target_format=param.get_path('learner.parameters.data.target_format')
                    )

                else:
                    validation_data_sequence = None

                # Get data item size
                data_size = train_data_sequence.data_size

            else:
                # Collect training data and corresponding targets to matrices
                log.line('Collecting training data', indent=2)

                X_train, Y_train, data_size = dcase_util.keras.data_collector(
                    item_list=item_list_train,
                    data_processing_chain=data_processing_chain,
                    meta_processing_chain=meta_processing_chain,
                    target_format=param.get_path('learner.parameters.data.target_format', 'single_target_per_sequence'),
                    channel_dimension=param.get_path('learner.parameters.data.data_format', 'channels_first'),
                    verbose=True,
                    print_indent=4
                )
                log.foot(indent=2)

                if item_list_validation:
                    log.line('Collecting validation data', indent=2)
                    X_validation, Y_validation, data_size = dcase_util.keras.data_collector(
                        item_list=item_list_validation,
                        data_processing_chain=data_processing_chain,
                        meta_processing_chain=meta_processing_chain,
                        target_format=param.get_path('learner.parameters.data.target_format', 'single_target_per_sequence'),
                        channel_dimension=param.get_path('learner.parameters.data.data_format', 'channels_first'),
                        verbose=True,
                        print_indent=4
                    )
                    log.foot(indent=2)

                    validation_data = (X_validation, Y_validation)

                else:
                    validation_data = None

            # Collect constants for the model generation, add class count and feature matrix size
            model_parameter_constants = {
                'CLASS_COUNT': int(db.scene_label_count()),
                'FEATURE_VECTOR_LENGTH': int(data_size['data']),
                'INPUT_SEQUENCE_LENGTH': int(data_size['time']),
            }

            # Read constants from parameters
            model_parameter_constants.update(
                param.get_path('learner.parameters.model.constants', {})
            )

            # Create sequential model
            keras_model = dcase_util.keras.create_sequential_model(
                model_parameter_list=param.get_path('learner.parameters.model.config'),
                constants=model_parameter_constants
            )

            # Create optimizer object
            param.set_path(
                path='learner.parameters.compile.optimizer',
                new_value=dcase_util.keras.create_optimizer(
                    class_name=param.get_path('learner.parameters.optimizer.class_name'),
                    config=param.get_path('learner.parameters.optimizer.config')
                )
            )

            # Compile model
            keras_model.compile(
                **param.get_path('learner.parameters.compile', {})
            )

            # Show model topology
            log.line(
                dcase_util.keras.model_summary_string(keras_model)
            )

            # Create callback list
            callback_list = [
                dcase_util.keras.ProgressLoggerCallback(
                    epochs=param.get_path('learner.parameters.fit.epochs'),
                    metric=param.get_path('learner.parameters.compile.metrics')[0],
                    loss=param.get_path('learner.parameters.compile.loss'),
                    output_type='logging'
                )
            ]

            if param.get_path('learner.parameters.callbacks.StopperCallback'):
                # StopperCallback
                callback_list.append(
                    dcase_util.keras.StopperCallback(
                        epochs=param.get_path('learner.parameters.fit.epochs'),
                        **param.get_path('learner.parameters.callbacks.StopperCallback', {})
                    )
                )

            if param.get_path('learner.parameters.callbacks.ProgressPlotterCallback'):
                # ProgressPlotterCallback
                callback_list.append(
                    dcase_util.keras.ProgressPlotterCallback(
                        epochs=param.get_path('learner.parameters.fit.epochs'),
                        **param.get_path('learner.parameters.callbacks.ProgressPlotterCallback', {})
                    )
                )

            if param.get_path('learner.parameters.callbacks.StasherCallback'):
                # StasherCallback
                callback_list.append(
                    dcase_util.keras.StasherCallback(
                        epochs=param.get_path('learner.parameters.fit.epochs'),
                        **param.get_path('learner.parameters.callbacks.StasherCallback', {})
                    )
                )

            if param.get_path('learner.parameters.callbacks.LearningRateWarmRestart'):
                # LearningRateWarmRestart
                callback_list.append(
                    dcase_util.keras.LearningRateWarmRestart(
                        nbatch=numpy.ceil(X_train.shape[0] / param.get_path('learner.parameters.fit.batch_size')),
                        **param.get_path('learner.parameters.callbacks.LearningRateWarmRestart', {})
                    )
                )

            # Train model
            if param.get_path('learner.parameters.generator.enable'):
                keras_model.fit_generator(
                    generator=train_data_sequence,
                    validation_data=validation_data_sequence,
                    callbacks=callback_list,
                    verbose=0,
                    epochs=param.get_path('learner.parameters.fit.epochs'),
                    shuffle=param.get_path('learner.parameters.fit.shuffle')
                )

            else:
                keras_model.fit(
                    x=X_train,
                    y=Y_train,
                    validation_data=validation_data,
                    callbacks=callback_list,
                    verbose=0,
                    epochs=param.get_path('learner.parameters.fit.epochs'),
                    batch_size=param.get_path('learner.parameters.fit.batch_size'),
                    shuffle=param.get_path('learner.parameters.fit.shuffle')
                )

            for callback in callback_list:
                if isinstance(callback, dcase_util.keras.StasherCallback):
                    # Fetch the best performing model
                    callback.log()
                    best_weights = callback.get_best()['weights']

                    if best_weights:
                        keras_model.set_weights(best_weights)

                    break

            # Save model
            keras_model.save(fold_model_filename)

            processed_files.append(fold_model_filename)

    return processed_files


def do_testing(db, scene_labels, folds, param, log, overwrite=False):
    """Testing stage

    Parameters
    ----------
    db : dcase_util.dataset.Dataset
        Dataset

    scene_labels : list of str
        List of scene labels

    folds : list
        List of active folds

    param : dcase_util.containers.DCASEAppParameterContainer
        Application parameters

    log : dcase_util.ui.FancyLogger
        Logging interface

    overwrite : bool
        Overwrite data always
        Default value False

    Returns
    -------
    list

    """

    processed_files = []

    # Loop over all cross-validation folds and test
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

        # Initialize model to None, load when first non-tested file encountered.
        keras_model = None

        # Create processing chain for features
        data_processing_chain = dcase_util.processors.ProcessingChain()
        for chain in param.get_path('data_processing_chain.parameters.chain'):
            processor_name = chain.get('processor_name')
            init_parameters = chain.get('init_parameters', {})

            # Inject parameters
            if processor_name == 'dcase_util.processors.NormalizationProcessor':
                # Get normalization factor filename
                init_parameters['filename'] = os.path.join(
                    param.get_path('path.application.feature_normalizer'),
                    'norm_fold_{fold}.cpickle'.format(fold=fold)
                )

            data_processing_chain.push_processor(
                processor_name=processor_name,
                init_parameters=init_parameters,
            )

        # Get results filename
        fold_results_filename = os.path.join(
            param.get_path('path.application.recognizer'),
            'res_fold_{fold}.csv'.format(fold=fold)
        )

        if not os.path.isfile(fold_results_filename) or overwrite:
            # Load model if not yet loaded
            if not keras_model:
                dcase_util.keras.setup_keras(
                    seed=param.get_path('learner.parameters.random_seed'),
                    profile=param.get_path('learner.parameters.keras_profile'),
                    backend=param.get_path('learner.parameters.backend', 'tensorflow'),
                    print_indent=2
                )
                import keras

                keras_model = keras.models.load_model(fold_model_filename)

            # Initialize results container
            res = dcase_util.containers.MetaDataContainer(
                filename=fold_results_filename
            )

            if not len(db.test(fold=fold)):
                raise ValueError('Dataset did not return any test files. Check dataset setup.')

            # Loop through all test files from the current cross-validation fold
            for item in db.test(fold=fold):
                # Get feature filename
                feature_filename = dcase_util.utils.Path(
                    path=item.filename
                ).modify(
                    path_base=param.get_path('path.application.feature_extractor'),
                    filename_extension='.cpickle'
                )

                features = data_processing_chain.process(
                    filename=feature_filename
                )
                input_data = features.data

                if len(keras_model.input_shape) == 4:
                    data_format = None
                    if isinstance(keras_model.get_config(), list):
                        data_format = keras_model.get_config()[0]['config']['data_format']
                    elif isinstance(keras_model.get_config(), dict) and 'layers' in keras_model.get_config():
                        data_format = keras_model.get_config()['layers'][0]['config']['data_format']

                    # Add channel
                    if data_format == 'channels_first':
                        input_data = numpy.expand_dims(input_data, 0)

                    elif data_format == 'channels_last':
                        input_data = numpy.expand_dims(input_data, 3)

                # Get network output
                probabilities = keras_model.predict(x=input_data).T

                if param.get_path('recognizer.collapse_probabilities.enable', True):
                    probabilities = dcase_util.data.ProbabilityEncoder().collapse_probabilities(
                        probabilities=probabilities,
                        operator=param.get_path('recognizer.collapse_probabilities.operator', 'sum'),
                        time_axis=1
                    )

                # Binarization of the network output
                frame_decisions = dcase_util.data.ProbabilityEncoder().binarization(
                    probabilities=probabilities,
                    binarization_type=param.get_path('recognizer.frame_binarization.type', 'global_threshold'),
                    threshold=param.get_path('recognizer.frame_binarization.threshold', 0.5)
                )

                estimated_scene_label = dcase_util.data.DecisionEncoder(
                    label_list=scene_labels
                ).majority_vote(
                    frame_decisions=frame_decisions
                )

                # Collect class wise probabilities and scale them between [0-1]
                class_probabilities = {}
                for scene_id, scene_label in enumerate(scene_labels):
                    class_probabilities[scene_label] = probabilities[scene_id] / input_data.shape[0] 

                res_data = {
                    'filename': db.absolute_to_relative_path(item.filename),
                    'scene_label': estimated_scene_label
                }
                # Add class class_probabilities
                res_data.update(class_probabilities)

                # Store result into results container
                res.append(
                    res_data
                )

                processed_files.append(item.filename)

            if not len(res):
                raise ValueError('No results to save.')

            # Save results container
            fields = ['filename', 'scene_label']
            fields += scene_labels

            res.save(fields=fields, csv_header=True)

    return processed_files


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

    devices = [
        'a',
        'b',
        'c',
        's1',
        's2',
        's3',
        's4',
        's5',
        's6'
    ]

    class_wise_results = numpy.zeros((1+len(devices), len(db.scene_labels())))
    fold = 1

    fold_results_filename = os.path.join(
        param.get_path('path.application.recognizer'),
        'res_fold_{fold}.csv'.format(fold=fold)
    )

    reference_scene_list = db.eval(fold=fold)

    reference_scene_list_devices  = {}
    for device in devices:
        reference_scene_list_devices[device] = dcase_util.containers.MetaDataContainer()

    for item_id, item in enumerate(reference_scene_list):
        device = os.path.splitext(os.path.split(item.filename)[-1])[0].split('-')[-1]

        reference_scene_list[item_id]['filename'] = os.path.split(item.filename)[-1]
        reference_scene_list[item_id]['file'] = item.filename

        reference_scene_list_devices[device].append(item)

    estimated_scene_list = dcase_util.containers.MetaDataContainer().load(
        filename=fold_results_filename,
        file_format=dcase_util.utils.FileFormat.CSV,
        csv_header=True,
        delimiter='\t'
    )

    estimated_scene_list_devices  = {}
    for device in devices:
        estimated_scene_list_devices[device] = dcase_util.containers.MetaDataContainer()

    for item_id, item in enumerate(estimated_scene_list):
        device = os.path.splitext(os.path.split(item.filename)[-1])[0].split('-')[-1]

        estimated_scene_list[item_id]['filename'] = os.path.split(item.filename)[-1]
        estimated_scene_list[item_id]['file'] = item.filename

        estimated_scene_list_devices[device].append(item)

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

    y_true_device = {}
    y_pred_device = {}

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

        if item.source_label not in y_true_device:
            y_true_device[item.source_label] = []
            y_pred_device[item.source_label] = []

        y_true_device[item.source_label].append(scene_label_id)
        y_pred_device[item.source_label].append(item_probabilities)

    from sklearn.metrics import log_loss
    logloss_overall = log_loss(y_true=y_true, y_pred=y_pred)

    logloss_class_wise = {}
    for scene_label in db.scene_labels():
        logloss_class_wise[scene_label] = log_loss(
            y_true=y_true_scene[scene_label],
            y_pred=y_pred_scene[scene_label],
            labels=list(range(len(db.scene_labels())))
        )

    logloss_device_wise = {}
    for decice_label in list(y_true_device.keys()):
        logloss_device_wise[decice_label] = log_loss(
            y_true=y_true_device[decice_label],
            y_pred=y_pred_device[decice_label],
            labels=list(range(len(db.scene_labels())))
        )

    results = evaluator.results()
    all_results.append(results)

    evaluator_devices = {}
    for device in devices:
        evaluator_devices[device] = sed_eval.scene.SceneClassificationMetrics(
            scene_labels=db.scene_labels()
        )

        evaluator_devices[device].evaluate(
            reference_scene_list=reference_scene_list_devices[device],
            estimated_scene_list=estimated_scene_list_devices[device]
        )

        results_device = evaluator_devices[device].results()
        all_results.append(results_device)

    for scene_label_id, scene_label in enumerate(db.scene_labels()):
        class_wise_results[0, scene_label_id] = results['class_wise'][scene_label]['accuracy']['accuracy']
        
        for device_id, device in enumerate(devices):
            class_wise_results[1+device_id, scene_label_id] = all_results[1+device_id]['class_wise'][scene_label]['accuracy']['accuracy']
            
    overall = [
        results['class_wise_average']['accuracy']['accuracy']
    ]
    for device_id, device in enumerate(devices):
        overall.append(all_results[1+device_id]['class_wise_average']['accuracy']['accuracy'])

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
    column_headers = ['Scene', 'Accuracy']
    column_widths = [16, 9]
    column_types = ['str20', 'float1_percentage']
    column_separators = [True, True]
    for dev_id, device in enumerate(devices):
        column_headers.append(device.upper())
        column_widths.append(8)
        column_types.append('float1')
        if dev_id < len(devices)-1:
            column_separators.append(False)
        else:
            column_separators.append(True)

    column_headers.append('Logloss')
    column_widths.append(10)
    column_types.append('float3')
    column_separators.append(False)

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
        row_data = [scene_label]
        for id in range(class_wise_results.shape[0]):
            row_data.append(class_wise_results[id, scene_label_id] * 100.0)
        row_data.append(logloss_class_wise[scene_label])
        log.row(*row_data)
    log.row_sep()

    # Last row
    column_values = ['Accuracy']
    for value in overall:
        column_values.append(value*100.0)
    column_values.append(' ')

    log.row(
        *column_values,
        types=column_types
    )

    column_values = ['Logloss', ' ']
    column_types = ['str20', 'float3']
    for device_label in devices:
        column_values.append(logloss_device_wise[device_label])
        column_types.append('float3')

    column_values.append(logloss_overall)
    column_types.append('float3')

    log.row(
        *column_values,
        types=column_types,
    )

    log.line()


if __name__ == "__main__":
    sys.exit(main(sys.argv))
