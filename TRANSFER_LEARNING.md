# Transfer Learning with GeoTransformer for RoboEye

This guide explains how to perform transfer learning using GeoTransformer, adapting pre-trained weights to RoboEye datasets with different voxel sizes and dataset formats.

## **1. System Architecture**

The transfer learning system provides a **production-ready workflow** with an **enhanced 5-phase strategic framework**:

### **1.1 Enhanced Strategic Framework**
- **5-Phase Progressive Strategy**: Scientifically designed phase transitions for optimal learning
- **Smart Checkpoint Management**: 80% storage reduction with intelligent saving
- **Advanced Training Features**: Multi-scale training, curriculum learning, adaptive loss weighting
- **Memory Optimization**: Selective checkpoint saving with automatic cleanup

### **1.2 Production Features**
- **Manual Phase Control**: Train each phase individually with full parameter control
- **Progressive Unfreezing**: Automatic epoch-based phase transitions for efficient learning
- **Dynamic Configuration**: No hardcoded dataset dependencies
- **Centralized Logic**: Single `setup_dataset_config()` function handles all dataset types
- **Clean Imports**: No side effects during module loading
- **Extensible Design**: Easy to add support for new dataset formats
- **Robust Error Handling**: Corrupted samples are skipped gracefully with detailed logging
- **Centralized Logging**: All components use unified logging with configurable output directories
- **Production Ready**: Comprehensive failure tracking and recovery mechanisms

### **1.3 Model Architecture Context**

The enhanced 5-phase strategy considers the complete model architecture:

#### **1.3.1 Model Components and Parameter Counts:**
- **backbone**: KPConvFPN (6,009,600 params) - 4 encoder + 3 decoder stages
- **transformer**: GeometricTransformer (3,819,776 params) - 6 alternating self/cross attention layers
- **coarse_target**: SuperPointTargetGenerator (0 params) - ALGORITHMIC
- **coarse_matching**: SuperPointMatching (0 params) - ALGORITHMIC
- **fine_matching**: LocalGlobalRegistration (0 params) - ALGORITHMIC
- **optimal_transport**: LearnableLogOptimalTransport (1 param) - LEARNABLE


## **2. Command-Line Arguments**

- `--dataset_id`: RoboEye dataset ID (e.g., "HST-YZ/CAD6608043632/2025-10-14-rvbust_RE_20_synthetic")
- `--dataset_root`: BOP dataset root path (e.g., "~/repos/roboeye/datasets/ITODD_converted")
- `--cache_bool`: Save train/validation dataset preprocessed or not (flag)
- `--train_split`: Data fraction for training [0-1] (default: 0.7)
- `--pretrained_weights`: Path to pretrained weights (.pth.tar)
- `--pretrained_voxel_size`: Voxel size of pretrained model (default: 0.025m)
- `--target_voxel_size`: Target voxel size (default: 0.009m)
- `--voxel_strategy`: Voxel size recommendation strategy (conservative/balanced/robust, default: conservative)
- `--phase`: Training phase (0-4, default: 0)
- `--enable_progressive_unfreezing`: Enable automatic progressive unfreezing (enhanced 5-phase strategy)
- `--freeze_epochs`: Custom freeze epoch schedule (default: "8 24 40 56")
- `--learning_rates`: Custom learning rate schedule for 5 phases (default: [1e-5, 5e-5, 1e-4, 5e-4, 1e-3])
- `--optim_max_epoch`: Number of epochs (default: 80)
- `--optim_lr`: Learning rate (overrides phase default)
- `--gradient_accumulation_steps`: Steps to accumulate gradients before optimizer update (default: 4)
- `--so3_curriculum_epochs`: Epochs to reach full SO3 rotation (default: 50)
- `--max_so3_rotation_deg`: Maximum SO3 rotation angle in degrees (default: 180.0)
- `--enable_so3_curriculum`: Enable SO3 curriculum learning (default: False)
- `--resume`: Resume from checkpoint (flag)
- `--continue_optimizer`: Continue optimizer state when resuming (same phase) (flag)
- `--output_dir`: Output directory (default: auto-generated)


## **3. Enhanced 5-Phase Strategic Framework**

### **3.1 Strategic Architecture Overview**

The enhanced framework implements a scientifically-designed progressive unfreezing strategy that addresses the core challenges of transfer learning with geometric deep learning models:

| Phase | Strategic Focus | Epochs | Trainable Components | Purpose |
|-------|----------------|---------|---------------------|---------|
| **Phase 0 (0)** | Scale Validation + Architectural Compatibility | 1-20 | `optimal_transport.alpha` + partial `geometric structure embedding` (1.3M param) | Scale parameter adaptation and architectural validation |
| **Phase A (1)** | Coarse Feature Adaptation + Early Attention | 21-40 | full `geometric structure embedding` + `self-attention`(s) (~3.8M, 38%) | Correspondence learning with adapted features |
| **Phase B (2)** | Cross-Modal Learning + Mid-Level Features | 41-60 | `input/output projections` + `transformer layers 0-3` (~5.4M, 54%) | Spatial-angular attention fine-tuning |
| **Phase C1 (3)** | Backbone Integration + Full Transformer | 61-80 | `transformer` + `backbone decoder` (~7.5M, 77%) | Progressive backbone unfreezing with preserved transformer |
| **Phase C2 (4)** | Full Model Polish + Domain Adaptation | 81-100 | Full model (~9.8M, 100%) | End-to-end optimization and domain polish |

**3.1.1 Parameter Scaling**
The system automatically scales parameters based on voxel size differences:
- **Scale Factor**: `target_voxel_size / pretrained_voxel_size`
- **Scaled Parameters**: Base radius, init radius, init sigma, sigma_d, sigma_a, acceptance radius, GT matching radius
- Example: Pretrained (2.5cm) → RoboEye (0.9cm) = Scale factor 0.36

**3.1.2 Expected Progressive Behavior**:
- **Phase 0**: NaN losses (expected with domain gap)
- **Phase A**: Transition to Phase 1 → losses become finite
- **Phase B**: Transition to Phase 2 → better feature quality
- **Phase C**: Transition to Phase 3 → optimal performance

**3.1.3 Metrics**
#### Loss Metrics
- `loss`: total loss of coarse and fine registration loss (latent space)
- `c_loss`: coarse registration loss (latent space)
- `f_loss`: fine registration loss (latent space)
#### Registration Accuracy Metrics
- `PIR`: position inlier ratio (phiscal world)
- `IR`: inlier ration (physcial world)
#### Registration Error Metrics
- `RRE`: registration rotational error
- `RTE`: registration translation error
- `RR`: total registration error

**3.1.4 Phase Success Criteria**
- **Phase 0 (0): Scale Validation + Architectural Compatibility**: Finite losses by end of phase (if using compatible voxel sizes)
- **Phase A (1): Coarse Feature Adaptation + Early Attention**: `PIR > 0.1`, finite coarse loss, improved correspondence quality
- **Phase B (2): Cross-Modal Learning + Mid-Level Features**: `RRE < 50°`, `IR > 0.1`, improved spatial understanding
- **Phase C1 (3): Backbone Integration + Full Transformer**: `RRE < 20°`, `RTE < 200mm`, enhanced feature quality
- **Phase C2 (4): Full Model Polish + Domain Adaptation**: `RRE < 10°`, `RTE < 100mm`, `RR > 0.1`


## **4. Implementation Details**

### **4.1 Progressive Unfreezing Implementation**

The system now supports two modes:

#### **4.1.1 Manual Phase Control (Default):**
```bash
# Phase 0: Scale validation
python train_transfer_learning.py \
  --dataset_id "HST-YZ/CAD6608043632/2025-10-14-rvbust_RE_20_synthetic" \
  --phase 0 --optim_max_epoch 8 \
  --cache_bool --train_split 0.7

# Phase 1: Coarse feature adaptation (resume from Phase 0)
python train_transfer_learning.py \
  --dataset_id "HST-YZ/CAD6608043632/2025-10-14-rvbust_RE_20_synthetic" \
  --phase 1 --optim_max_epoch 24 \
  --resume --continue_optimizer

# Phase 2: Cross-modal learning
python train_transfer_learning.py \
  --dataset_id "HST-YZ/CAD6608043632/2025-10-14-rvbust_RE_20_synthetic" \
  --phase 2 --optim_max_epoch 40 \
  --resume --continue_optimizer \
  --gradient_accumulation_steps 4
```

#### **4.1.2 Automatic Progressive Unfreezing:**
```bash
# Enable progressive unfreezing with custom schedule
python train_transfer_learning.py --enable_progressive_unfreezing \
  --gradient_accumulation_steps 4 \
  --freeze_epochs "8 24 40 56" \
  --learning_rates 1e-5 5e-5 1e-4 5e-4 1e-3 \
  --enable_progressive_unfreezing \
  --optim_max_epoch 80 \
  --enable_so3_curriculum \
  --so3_curriculum_epochs 50
```

#### **4.1.3 Automatic Phase Detection:**
The system automatically detects and resumes from previous phases:
- Looks for `best_phase_{N-1}.pth.tar` when starting phase N
- Falls back to `latest_phase_{N-1}.pth.tar` if best not found
- Logs warnings if no previous phase checkpoint available


#### **4.1.4 When to Use Progressive Unfreezing**
##### **Good for**:
- Research and experimentation
- Small datasets with fast iterations
- Exploring optimal transition timing
- Domain gap analysis

##### **Avoid for**:
- Production training (use manual phases)
- When you need precise control over each phase
- Resuming from specific checkpoints


### **4.2 SO3 Curriculum Learning**
- progressive rotation augmentation:
```bash
# Enable SO3 curriculum learning
python train_transfer_learning.py --enable_so3_curriculum \
  --so3_curriculum_epochs 50 \
  --max_so3_rotation_deg 180.0

# Custom curriculum schedule
python train_transfer_learning.py --enable_so3_curriculum \
  --so3_curriculum_epochs 30 \
  --max_so3_rotation_deg 90.0
```
### **4.3 Gradient Accumulation**
Support for training with large effective batch sizes:
```bash
# Use gradient accumulation for memory efficiency
python train_transfer_learning.py --gradient_accumulation_steps 4

# Combined with curriculum learning
python train_transfer_learning.py --enable_so3_curriculum \
  --gradient_accumulation_steps 8 \
  --batch_size 1
```
### **4.4 Memory Optimization**
- **Automatic Cleanup**: Removes old checkpoints when limits exceeded
- **Size Monitoring**: Tracks storage usage and maintains configurable limits
- **Descriptive Naming**: Checkpoints include phase, voxel size, and key parameters
- **Resume Capability**: Easy resumption from best or phase-specific checkpoints

### **4.5 Curriculum Learning Integration**
- **Multi-Scale Training Progression**: Gradual increase in geometric complexity
- **Feature Complexity Scheduling**: Progressive unfreezing preserves learned representations
- **Adaptive Loss Weighting**: Dynamic balancing of coarse vs fine matching losses

### **4.6 Robustness Mechanisms**
- **Conservative Learning Rates**: Phase-specific rates prevent catastrophic forgetting
- **Gradual Domain Adaptation**: Progressive feature adaptation reduces training instability
- **Validation-Based Best Model Selection**: Automatic detection and preservation of optimal checkpoints

### **4.7 Corrupted Sample Handling**
- **Automatic Skip**: Corrupted or missing samples are automatically skipped
- **Detailed Logging**: Each failure is logged with sample index, name, dataset_id, and error details
- **Failure Reports**: JSON logs are generated for analysis: `{subset}_dataset_failures.log`
- **Graceful Recovery**: Training continues with valid samples only

### **4.8 Centralized Logging**
- **Single Log Directory**: All logs go to `output/transfer_learning_phase_{N}/logs/`
- **Structured Output**: Consistent log format across all components
- **File + Console**: Logs written to both files and displayed on console
- **Failure Tracking**: Dataset failures logged to separate JSON files for analysis

### **4.9 Smart Saving Features:**
- **Selective Frequency**: Default save every 25 epochs (configurable via `--save_frequency`)
- **Best Model Detection**: Automatic tracking with `best_val_loss` and `best_val_epoch`
- **Phase Completion Snapshots**: Special checkpoints saved at phase completion
- **Storage Optimization**: Automatic cleanup maintains `max_checkpoints` limit

### **4.10 Checkpoint Types:**
1. **Best Checkpoints**: `best_phase_{N}.pth.tar` - Best validation performance
2. **Latest Phase Checkpoints**: `latest_phase_{N}.pth.tar` - Most recent for each phase
3. **Phase Completion Checkpoints**: `phase_{N}_complete_voxel{size}mm_scale{factor}.pth.tar`
4. **Regular Checkpoints**: `checkpoint_epoch_{epoch}_phase_{phase}_{reason}_voxel{size}mm_*.pth.tar`

### **4.11 Automatic Cleanup:**
- Removes old checkpoints when exceeding `max_checkpoints` limit
- Maintains phase-specific checkpoints for easy resumption
- Cleans up epoch-specific files to prevent storage bloat


## **5. MISC**

### **5.1 Best Practices**
1. **Always test first**: Use `test_transfer_learning.py` before training
2. **Start with defaults**: Use default pretrained weights **withotu** specifying `--pretrained_weights` for initial experiments

### **5.2 Configuration Details**
The transfer learning configuration automatically:
- Scales all spatial parameters based on voxel size ratio
- Sets up progressive learning rates for each phase
- Configures appropriate data loaders for each dataset format

### **5.3 Files Overview**
- `train_transfer_learning.py` - Main script. Single phase training with comprehensive logging
- `test_transfer_learning.py` - Setup validation and model evaluation for both dataset formats
- `inspect_model.py` - Model architecture inspection utility
- `utils.py` - Configuration and model setup utility functions

### **5.4 Configuration Files**
- `experiments/roboeye/config.py` - **Refactored**: Dynamic configuration with `setup_dataset_config()`
- `experiments/roboeye/dataset.py` - **Updated**: Data loaders creation


### **5.5 Default Pretrained Weights**

The system looks for default pretrained weights at:
```
~/repos/roboeye/iris/tools/models/pretrained/geotransformer/geotransformer-3dmatch.pth.tar
```

### **5.6 Output Structure**
```
output/
├── transfer_learning_phase_0/          # Phase 0 output
│   ├── logs/
│   │   ├── train.log                   # Training progress
│   │   └── failures.log                # Dataset failures
│   ├── snapshots/
│   │   ├── best_phase_0.pth.tar        # Best validation checkpoint
│   │   ├── latest_phase_0.pth.tar      # Latest checkpoint
│   │   ├── phase_0_complete_*.pth.tar  # Phase completion
│   │   └── checkpoint_epoch_*_*.pth.tar # Regular checkpoints
│   └── tensorboard/                    # TensorBoard logs
├── transfer_learning_phase_1/
├── transfer_learning_phase_2/
├── transfer_learning_phase_3/
└── transfer_learning_phase_4/
```
**5.6.1 Checkpoint Naming Convention**:
`checkpoint_epoch_{epoch}_phase_{phase}_voxel{voxel*1000}mm_sigma-d{sigma_d*1000}mm_sigma-a{sigma_a}_angle-k{angle_k}_scale{scale_factor*1000}.pth.tar`


### **5.7 Common Issues**
1. **Dataset not found**: Verify dataset_id exists in Nova factory or dataset_root path is correct
2. **CUDA memory issues**: Reduce batch size with `--batch_size 1`
3. **Missing pretrained weights**: Ensure default path exists or specify `--pretrained_weights`
4. **Configuration errors**: Use `test_transfer_learning.py` to validate pipeline functionality before training

#### **5.7.1 Debugging**
Use the test script to validate setup before training:
```bash
python test_transfer_learning.py --help
```

## **6. Training Lauch and Management System**

The Training Management System provides a complete solution for launching, monitoring, and managing GeoTransformer training sessions with production-grade features:

- **Launch Scripts**: Intelligent GPU selection and process management
- **Monitoring Tools**: Real-time training progress and resource monitoring
- **Process Control**: Safe termination and cleanup
- **Log Management**: Automated log cleaning and viewing

### **6.1. Training Launch Scripts**
#### **6.1.1 `launch_training_monitored.sh` - General Purpose Launch**
- **Auto GPU Selection**: Picks least-utilized GPU
- **Memory Optimization**: CUDA memory configuration
- **Background/Foreground Modes**: Flexible operation modes
- **Process Monitoring**: Integrated with `process_debugger.sh`

#### **6.1.2 Usage:**
```bash
# Auto GPU selection (background)
./launch_training_monitored.sh

# Specific GPU
./launch_training_monitored.sh 1

# Foreground mode (systemd compatible)
./launch_training_monitored.sh --foreground
```

---

### **6.2 Monitoring Tools**
#### **6.2.1 `monitor_training.sh` - Live Training Monitor**
- **Real-time Monitoring**: Auto-refreshing dashboard
- **GPU Memory Tracking**: Live GPU utilization
- **Training Progress**: Epoch/loss information
- **System Load**: CPU/memory monitoring

#### **6.2.2 Usage:**
```bash
# Default 10-second refresh
./monitor_training.sh

# Custom refresh interval (5 seconds)
./monitor_training.sh 5

# Monitor output shows:
# ✓ Process status and PID
# ✓ GPU memory usage per device
# ✓ Training progress (epochs, losses)
# ✓ System load and cache status
```

#### **6.2.3 Dashboard Example:**
```
==========================================
Training Monitor - 2024-01-15 14:30:00
==========================================

✓ Training PID: 12345

[Process Memory]
  Current RSS: 4.25 GB
  Peak RSS:    6.10 GB

[GPU Memory]
  GPU 0: 7.2/8.0 GB (90%, Util: 95%)
  Training Process: 5.8 GB

[Training Progress]
  Epoch: 12/80, Loss: 0.45, RRE: 8.7°, RTE: 24mm

[System Load]
  load average: 2.1, 1.8, 1.5
```

---

### **6.3 `process_logs_debugger.sh` - Log Viewer and Cleaner**
- **Log Cleaning**: Automatically removes noisy lines
- **Multi-log Tail**: Follows all relevant log files
- **Process Discovery**: Finds and shows running training processes

#### **6.3.1 Usage:**
```bash
# Auto-detect and monitor
./process_logs_debugger.sh

# Monitor specific PID
./process_logs_debugger.sh 12345

# Log cleaning removes:
# "Loading directly from disk as overwrite is set to False..."
```

### **6.4 Process Control**

#### **6.4.1 `kill_training.sh` - Safe Process Termination**
- **Graceful Shutdown**: SIGTERM first, SIGKILL if needed
- **PID File Management**: Cleans up PID files
- **Stray Process Cleanup**: Finds orphaned training processes
- **Force Option**: `--force` for immediate termination

#### **6.4.2 Usage:**
```bash
# Graceful termination
./kill_training.sh

# Force immediate kill
./kill_training.sh --force

# Output example:
✓ Training stopped
✓ Debugger stopped
✓ No stray processes found
```

### **6.5 Complete Workflow Example**

#### **6.5.1 Start Training Session**
```bash
# Step 1: Launch training
./launch_training_monitored.sh

# Output:
✓ Training launched in background on GPU 1
✓ Launcher PID: 5678
✓ Training PID will be in: process_debugger.pid
```

#### **6.5.2 Monitor Training**
```bash
# Option A: Live dashboard (recommended)
./monitor_training.sh

# Option B: Tail logs directly
./process_logs_debugger.sh

# Option C: Check GPU status
watch -n 2 nvidia-smi

# Option D: Quick status check
ps aux | grep train_transfer_learning
```

#### **6.5.3 Check Training Health**
```bash
# Check if training is running
if [ -f process_debugger.pid ]; then
    PID=$(cat process_debugger.pid)
    if kill -0 $PID 2>/dev/null; then
        echo "Training running (PID: $PID)"
    else
        echo "Training stopped unexpectedly"
    fi
fi
```

#### **6.5.4 Stop Training**
```bash
# Graceful stop
./kill_training.sh

# If stuck, force stop
./kill_training.sh --force

# Manual cleanup if needed
rm -f process_debugger.pid process_debugger.launcher.pid
```

### **6.6 Configuration Guide**

#### **6.6.1 Memory Settings**
```bash
# In launch scripts:
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128"

# Adjust based on GPU:
# 8GB GPU  → max_split_size_mb:64
# 16GB GPU → max_split_size_mb:128
# 24GB GPU → max_split_size_mb:256
```

#### **6.6.2 Training Parameters**
```bash
# Customize in launch scripts:
EPOCHS=150
FREEZE_EPOCHS="5 25 50 75"
DATASET_ID="your/dataset/id"
```

#### **6.6.3 Log Management**
```bash
# Log files created:
train_run_YYYYMMDD_HHMMSS.log      # Training output
process_debugger_YYYYMMDD_HHMMSS.out # Debugger output

# Clean logs automatically:
./process_logs_debugger.sh         # Removes noisy lines
```

### **6.7. Process Troubleshooting Guide**

#### **6.7.1 Common Issues**

##### **6.7.1.1 Issue: Training Dies Immediately**
```bash
# Check logs
tail -n 100 process_debugger_*.out

# Check GPU memory
nvidia-smi

# Common fixes:
# 1. Reduce batch size
# 2. Lower learning rate
# 3. Check dataset path
```

##### **6.7.1.2 Issue: Can't Find PID Files**
```bash
# Manual process discovery
pgrep -f "train_transfer_learning.py"

# Check all training processes
ps aux | grep -E "(bazel|python.*train)"

# Clean up manually
pkill -f "train_transfer_learning.py"
```

##### **6.7.1.3 Issue: GPU Memory Full**
```bash
# Kill all CUDA processes
nvidia-smi --query-compute-apps=pid --format=csv,noheader | xargs kill -9

# Reset GPU (careful!)
sudo nvidia-smi --gpu-reset
```

#### **6.7.2 Performance Issues**
```bash
# Check GPU utilization
watch -n 1 nvidia-smi

# Monitor CPU usage
top -u $(whoami)

# Check disk I/O
iotop -o
```

### **6.8. Process Best Practices**

#### **6.8.1 Pre-flight Checklist**
```bash
# Before training:
✓ Check GPU availability: nvidia-smi
✓ Verify dataset access: ls dataset/
✓ Ensure disk space: df -h .
✓ Test with small run: --epochs 1 --batch_size 1
```

#### **6.8.2 Monitoring Strategy**
```bash
# Regular checks:
# 1. GPU temperature (<85°C)
# 2. Memory usage (not >90%)
# 3. Checkpoint creation (every ~30min)
# 4. Loss convergence (should decrease)
```

#### **6.8.3 Cleanup Procedures**
```bash
# After training completes:
# 1. Archive logs
tar -czf training_$(date +%Y%m%d).tar.gz *.log *.out

# 2. Clean PID files
rm -f process_debugger*.pid

# 3. Backup checkpoints
cp -r output/transfer_learning_* backup/

# 4. Clear cache if needed
rm -rf output/*/cache/
```


### **6.9 Performance Monitoring**
```bash
# Add to monitoring script:
if [ $(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader) -gt 85 ]; then
    echo "ALERT: GPU overheating!"
fi

if [ $(df . --output=pcent | tail -1 | tr -d '%') -gt 90 ]; then
    echo "ALERT: Disk space low!"
fi
```

### **6.10 Quick Reference**
```bash
# Start training
./launch_training_monitored.sh

# Monitor
./monitor_training.sh

# View logs
./process_logs_debugger.sh

# Stop training
./kill_training.sh

# Force stop
./kill_training.sh --force

# Check status
ps aux | grep train_transfer_learning
```

#### **6.10.1 File Locations**
```bash
# PID files
process_debugger.pid              # Training process
process_debugger.launcher.pid     # Launcher process

# Log files
train_run_*.log                   # Training output
process_debugger_*.out           # Debugger output
output/*/logs/train-*.log        # Phase logs

# Checkpoints
output/transfer_learning_phase_*/snapshots/*.pth.tar
```

#### **6.10.2 Environment Variables**
```bash
# Set in launch scripts:
CUDA_VISIBLE_DEVICES              # GPU selection
PYTORCH_CUDA_ALLOC_CONF          # Memory config
CUDA_LAUNCH_BLOCKING             # Debug mode
```
