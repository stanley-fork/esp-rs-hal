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
        // see <https://github.com/espressif/esp-idf/blob/dc016f5987/components/hal/esp32c6/include/hal/efuse_ll.h#L65-L73>
        // <https://github.com/espressif/esp-idf/blob/903af13e8/components/efuse/esp32c6/esp_efuse_table.csv#L156>
        (
            Self::read_field_le::<u8>(BLK_VERSION_MAJOR),
            Self::read_field_le::<u8>(BLK_VERSION_MINOR),
        )
    }

    /// Get version of RTC calibration block
    ///
    /// see <https://github.com/espressif/esp-idf/blob/903af13e8/components/efuse/esp32c6/esp_efuse_rtc_calib.c#L20>
    pub fn rtc_calib_version() -> u8 {
        let (_major, minor) = Self::block_version();
        if minor >= 1 { 1 } else { 0 }
    }

    /// Get ADC initial code for specified attenuation from efuse
    ///
    /// see <https://github.com/espressif/esp-idf/blob/903af13e8/components/efuse/esp32c6/esp_efuse_rtc_calib.c#L32>
    pub fn rtc_calib_init_code(_unit: u8, atten: Attenuation) -> Option<u16> {
        let version = Self::rtc_calib_version();

        if version != 1 {
            return None;
        }

        // See <https://github.com/espressif/esp-idf/blob/903af13e8/components/efuse/esp32c6/esp_efuse_table.csv#L147-L152>
        let init_code: u16 = Self::read_field_le(match atten {
            Attenuation::_0dB => ADC1_INIT_CODE_ATTEN0,
            Attenuation::_2p5dB => ADC1_INIT_CODE_ATTEN1,
            Attenuation::_6dB => ADC1_INIT_CODE_ATTEN2,
            Attenuation::_11dB => ADC1_INIT_CODE_ATTEN3,
        });

        Some(init_code + 1600) // version 1 logic
    }

    /// Get ADC reference point voltage for specified attenuation in millivolts
    ///
    /// see <https://github.com/espressif/esp-idf/blob/903af13e8/components/efuse/esp32c6/esp_efuse_rtc_calib.c#L42>
    pub fn rtc_calib_cal_mv(_unit: u8, atten: Attenuation) -> u16 {
        match atten {
            Attenuation::_0dB => 400,
            Attenuation::_2p5dB => 550,
            Attenuation::_6dB => 750,
            Attenuation::_11dB => 1370,
        }
    }

    /// Get ADC reference point digital code for specified attenuation
    ///
    /// see <https://github.com/espressif/esp-idf/blob/903af13e8/components/efuse/esp32c6/esp_efuse_rtc_calib.c#L42>
    pub fn rtc_calib_cal_code(_unit: u8, atten: Attenuation) -> Option<u16> {
        let version = Self::rtc_calib_version();

        if version != 1 {
            return None;
        }

        // See <https://github.com/espressif/esp-idf/blob/903af13e8/components/efuse/esp32c6/esp_efuse_table.csv#L153-L156>
        let cal_code: u16 = Self::read_field_le(match atten {
            Attenuation::_0dB => ADC1_CAL_VOL_ATTEN0,
            Attenuation::_2p5dB => ADC1_CAL_VOL_ATTEN1,
            Attenuation::_6dB => ADC1_CAL_VOL_ATTEN2,
            Attenuation::_11dB => ADC1_CAL_VOL_ATTEN3,
        });

        let cal_code = if cal_code & (1 << 9) != 0 {
            1500 - (cal_code & !(1 << 9))
        } else {
            1500 + cal_code
        };

        Some(cal_code)
    }

    /// Returns the major hardware revision
    pub fn major_chip_version() -> u8 {
        Self::read_field_le(WAFER_VERSION_MAJOR)
    }

    /// Returns the minor hardware revision
    pub fn minor_chip_version() -> u8 {
        Self::read_field_le(WAFER_VERSION_MINOR)
    }

    /// Returns the hardware revision
    ///
    /// The chip version is calculated using the following
    /// formula: MAJOR * 100 + MINOR. (if the result is 1, then version is v0.1)
    pub fn chip_revision() -> u16 {
        Self::major_chip_version() as u16 * 100 + Self::minor_chip_version() as u16
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
