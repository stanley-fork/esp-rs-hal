use crate::{analog::adc::Attenuation, peripherals::EFUSE};

mod fields;
pub use fields::*;

impl super::Efuse {
    /// Get status of SPI boot encryption.
    pub fn flash_encryption() -> bool {
        !Self::read_field_le::<u8>(SPI_BOOT_CRYPT_CNT)
            .count_ones()
            .is_multiple_of(2)
    }

    /// Get the multiplier for the timeout value of the RWDT STAGE 0 register.
    pub fn rwdt_multiplier() -> u8 {
        Self::read_field_le::<u8>(WDT_DELAY_SEL)
    }

    /// Get efuse block version
    ///
    /// see <https://github.com/espressif/esp-idf/blob/dc016f5987/components/hal/efuse_hal.c#L27-L30>
    pub fn block_version() -> (u8, u8) {
        // see <https://github.com/espressif/esp-idf/blob/dc016f5987/components/hal/esp32s3/include/hal/efuse_ll.h#L65-L73>
        // <https://github.com/espressif/esp-idf/blob/903af13e8/components/efuse/esp32s3/esp_efuse_table.csv#L196>
        (
            Self::read_field_le::<u8>(BLK_VERSION_MAJOR),
            Self::read_field_le::<u8>(BLK_VERSION_MINOR),
        )
    }

    /// Get version of RTC calibration block
    ///
    /// see <https://github.com/espressif/esp-idf/blob/903af13e8/components/efuse/esp32s3/esp_efuse_rtc_calib.c#L15>
    pub fn rtc_calib_version() -> u8 {
        let (major, _minor) = Self::block_version();

        if major == 1 { 1 } else { 0 }
    }

    /// Get ADC initial code for specified attenuation from efuse
    ///
    /// see <https://github.com/espressif/esp-idf/blob/903af13e8/components/efuse/esp32s3/esp_efuse_rtc_calib.c#L28>
    pub fn rtc_calib_init_code(unit: u8, atten: Attenuation) -> Option<u16> {
        let version = Self::rtc_calib_version();

        if version != 1 {
            return None;
        }

        let adc_icode_diff: [u16; 4] = if unit == 0 {
            [
                Self::read_field_le(ADC1_INIT_CODE_ATTEN0),
                Self::read_field_le(ADC1_INIT_CODE_ATTEN1),
                Self::read_field_le(ADC1_INIT_CODE_ATTEN2),
                Self::read_field_le(ADC1_INIT_CODE_ATTEN3),
            ]
        } else {
            [
                Self::read_field_le(ADC2_INIT_CODE_ATTEN0),
                Self::read_field_le(ADC2_INIT_CODE_ATTEN1),
                Self::read_field_le(ADC2_INIT_CODE_ATTEN2),
                Self::read_field_le(ADC2_INIT_CODE_ATTEN3),
            ]
        };

        // Version 1 logic for calculating ADC ICode based on EFUSE burnt value

        let mut adc_icode = [0; 4];
        if unit == 0 {
            adc_icode[0] = adc_icode_diff[0] + 1850;
            adc_icode[1] = adc_icode_diff[1] + adc_icode[0] + 90;
            adc_icode[2] = adc_icode_diff[2] + adc_icode[1];
            adc_icode[3] = adc_icode_diff[3] + adc_icode[2] + 70;
        } else {
            adc_icode[0] = adc_icode_diff[0] + 2020;
            adc_icode[1] = adc_icode_diff[1] + adc_icode[0];
            adc_icode[2] = adc_icode_diff[2] + adc_icode[1];
            adc_icode[3] = adc_icode_diff[3] + adc_icode[2];
        }

        Some(
            adc_icode[match atten {
                Attenuation::_0dB => 0,
                Attenuation::_2p5dB => 1,
                Attenuation::_6dB => 2,
                Attenuation::_11dB => 3,
            }],
        )
    }

    /// Get ADC reference point voltage for specified attenuation in millivolts
    ///
    /// see <https://github.com/espressif/esp-idf/blob/903af13e8/components/efuse/esp32s3/esp_efuse_rtc_calib.c#L63>
    pub fn rtc_calib_cal_mv(_unit: u8, _atten: Attenuation) -> u16 {
        850
    }

    /// Get ADC reference point digital code for specified attenuation
    ///
    /// see <https://github.com/espressif/esp-idf/blob/903af13e8/components/efuse/esp32s3/esp_efuse_rtc_calib.c#L63>
    pub fn rtc_calib_cal_code(unit: u8, atten: Attenuation) -> Option<u16> {
        let version = Self::rtc_calib_version();

        if version != 1 {
            return None;
        }

        let adc_vol_diff: [u16; 8] = [
            Self::read_field_le(ADC1_CAL_VOL_ATTEN0),
            Self::read_field_le(ADC1_CAL_VOL_ATTEN1),
            Self::read_field_le(ADC1_CAL_VOL_ATTEN2),
            Self::read_field_le(ADC1_CAL_VOL_ATTEN3),
            Self::read_field_le(ADC2_CAL_VOL_ATTEN0),
            Self::read_field_le(ADC2_CAL_VOL_ATTEN1),
            Self::read_field_le(ADC2_CAL_VOL_ATTEN2),
            Self::read_field_le(ADC2_CAL_VOL_ATTEN3),
        ];

        let mut adc1_vol = [0; 4];
        let mut adc2_vol = [0; 4];
        adc1_vol[3] = adc_vol_diff[3] + 900;
        adc1_vol[2] = adc_vol_diff[2] + adc1_vol[3] + 800;
        adc1_vol[1] = adc_vol_diff[1] + adc1_vol[2] + 700;
        adc1_vol[0] = adc_vol_diff[0] + adc1_vol[1] + 800;
        adc2_vol[3] = adc1_vol[3] - adc_vol_diff[7] + 15;
        adc2_vol[2] = adc1_vol[2] - adc_vol_diff[6] + 20;
        adc2_vol[1] = adc1_vol[1] - adc_vol_diff[5] + 10;
        adc2_vol[0] = adc1_vol[0] - adc_vol_diff[4] + 40;

        let atten = match atten {
            Attenuation::_0dB => 0,
            Attenuation::_2p5dB => 1,
            Attenuation::_6dB => 2,
            Attenuation::_11dB => 3,
        };

        Some(if unit == 0 {
            adc1_vol[atten]
        } else {
            adc2_vol[atten]
        })
    }
}

#[derive(Debug, Clone, Copy, strum::FromRepr)]
#[repr(u32)]
pub(crate) enum EfuseBlock {
    Block0,
    Block1,
    Block2,
    Block3,
    Block4,
    Block5,
    Block6,
    Block7,
    Block8,
    Block9,
    Block10,
}

impl EfuseBlock {
    pub(crate) fn address(self) -> *const u32 {
        let efuse = EFUSE::regs();
        match self {
            Self::Block0 => efuse.rd_wr_dis().as_ptr(),
            Self::Block1 => efuse.rd_mac_spi_sys_0().as_ptr(),
            Self::Block2 => efuse.rd_sys_part1_data0().as_ptr(),
            Self::Block3 => efuse.rd_usr_data0().as_ptr(),
            Self::Block4 => efuse.rd_key0_data0().as_ptr(),
            Self::Block5 => efuse.rd_key1_data0().as_ptr(),
            Self::Block6 => efuse.rd_key2_data0().as_ptr(),
            Self::Block7 => efuse.rd_key3_data0().as_ptr(),
            Self::Block8 => efuse.rd_key4_data0().as_ptr(),
            Self::Block9 => efuse.rd_key5_data0().as_ptr(),
            Self::Block10 => efuse.rd_sys_part2_data0().as_ptr(),
        }
    }
}
